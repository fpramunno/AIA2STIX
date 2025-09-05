#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Augmentation for AIA2STIX Dataset

This module implements data augmentation techniques that preserve the relationship
between AIA images and their corresponding STIX visibilities. Focus on spatial
transformations and noise without changing intensity distributions.

Key augmentation strategies:
1. Gaussian Blur: Preserve visibilities (mild spatial changes)
2. Noise Addition: Preserve visibilities (spatial structure unchanged)  
3. Small Rotations: Preserve visibilities (minimal spatial changes)
4. Elastic Deformations: Preserve visibilities (local spatial changes)
5. Multiple crops from same source: Same visibilities (same flare region)

@author: francesco
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import random
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter, rotate
from skimage.transform import rotate as skimage_rotate
from skimage.util import random_noise
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class AIA2STIXAugmentor:
    """
    Augmentation class for AIA2STIX image-visibility pairs.
    
    Focus on spatial transformations that don't alter intensity distributions:
    - Gaussian blur
    - Additive noise
    - Small rotations
    - Elastic deformations
    """
    
    def __init__(
        self, 
        rotation_range: Tuple[float, float] = (-15, 15),  # degrees
        noise_std_range: Tuple[float, float] = (0.0, 0.05),
        blur_sigma_range: Tuple[float, float] = (0.0, 1.5),
        elastic_alpha_range: Tuple[float, float] = (0, 50),
        elastic_sigma_range: Tuple[float, float] = (4, 8),
        augmentation_prob: float = 0.8
    ):
        """
        Initialize augmentor with parameter ranges.
        
        Args:
            rotation_range: (min, max) rotation angle in degrees
            noise_std_range: (min, max) Gaussian noise standard deviation
            blur_sigma_range: (min, max) Gaussian blur sigma
            elastic_alpha_range: (min, max) elastic deformation strength
            elastic_sigma_range: (min, max) elastic deformation smoothness
            augmentation_prob: Probability of applying augmentation
        """
        self.rotation_range = rotation_range
        self.noise_std_range = noise_std_range
        self.blur_sigma_range = blur_sigma_range
        self.elastic_alpha_range = elastic_alpha_range
        self.elastic_sigma_range = elastic_sigma_range
        self.augmentation_prob = augmentation_prob
    
    def apply_rotation(
        self, 
        image: np.ndarray, 
        angle: Optional[float] = None,
        preserve_shape: bool = True
    ) -> np.ndarray:
        """Apply small rotation to preserve spatial relationships."""
        if angle is None:
            angle = random.uniform(*self.rotation_range)
        
        if angle == 0:
            return image
        
        # Rotate with same background value (minimum of image)
        rotated = skimage_rotate(
            image, 
            angle, 
            resize=not preserve_shape,
            center=None,  # Center of image
            order=1,  # Linear interpolation
            mode='constant',
            cval=np.min(image),  # Use minimum value as background
            preserve_range=True
        )
        
        return rotated.astype(image.dtype)
    
    def apply_gaussian_noise(
        self, 
        image: np.ndarray, 
        noise_std: Optional[float] = None
    ) -> np.ndarray:
        """Add Gaussian noise scaled to image intensity."""
        if noise_std is None:
            noise_std = random.uniform(*self.noise_std_range)
        
        if noise_std == 0:
            return image
        
        # Scale noise relative to image standard deviation
        image_std = np.std(image)
        noise_level = noise_std * image_std
        noise = np.random.normal(0, noise_level, image.shape)
        
        # Add noise and ensure non-negative values
        noisy_image = image + noise
        return np.maximum(noisy_image, 0).astype(image.dtype)
    
    def apply_gaussian_blur(
        self, 
        image: np.ndarray, 
        sigma: Optional[float] = None
    ) -> np.ndarray:
        """Apply Gaussian blur."""
        if sigma is None:
            sigma = random.uniform(*self.blur_sigma_range)
        
        if sigma == 0:
            return image
        
        return gaussian_filter(image, sigma=sigma).astype(image.dtype)
    
    def apply_elastic_deformation(
        self,
        image: np.ndarray,
        alpha: Optional[float] = None,
        sigma: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply elastic deformation to the image.
        
        Args:
            image: Input image
            alpha: Deformation strength
            sigma: Deformation smoothness
            random_state: Random seed for reproducibility
        """
        if alpha is None:
            alpha = random.uniform(*self.elastic_alpha_range)
        if sigma is None:
            sigma = random.uniform(*self.elastic_sigma_range)
            
        if alpha == 0:
            return image
        
        if random_state is not None:
            np.random.seed(random_state)
        
        shape = image.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply deformation
        deformed = map_coordinates(image, indices, order=1, mode='reflect')
        return deformed.reshape(shape).astype(image.dtype)
    
    def apply_random_crop_and_resize(
        self,
        image: np.ndarray,
        crop_fraction_range: Tuple[float, float] = (0.8, 0.95),
        target_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply random crop and resize back to original size.
        This simulates slight zoom variations.
        """
        if target_size is None:
            target_size = min(image.shape)
        
        crop_fraction = random.uniform(*crop_fraction_range)
        crop_size = int(target_size * crop_fraction)
        
        h, w = image.shape
        if crop_size >= min(h, w):
            return image
        
        # Random crop location
        start_h = random.randint(0, h - crop_size)
        start_w = random.randint(0, w - crop_size)
        
        cropped = image[start_h:start_h + crop_size, start_w:start_w + crop_size]
        
        # Resize back using scipy
        from scipy.ndimage import zoom
        zoom_factor = target_size / crop_size
        resized = zoom(cropped, zoom_factor, order=1)
        
        # Ensure exact target size
        if resized.shape[0] != target_size or resized.shape[1] != target_size:
            # Center crop/pad to exact size
            curr_h, curr_w = resized.shape
            if curr_h > target_size:
                start_h = (curr_h - target_size) // 2
                resized = resized[start_h:start_h + target_size, :]
            if curr_w > target_size:
                start_w = (curr_w - target_size) // 2
                resized = resized[:, start_w:start_w + target_size]
        
        return resized.astype(image.dtype)
    
    def apply_horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply horizontal flip."""
        return np.fliplr(image).copy()
    
    def apply_vertical_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply vertical flip."""
        return np.flipud(image).copy()
    
    def augment_image_visibility_pair(
        self, 
        image: np.ndarray, 
        visibilities: np.ndarray,
        num_augmentations: int = 4
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate multiple augmented versions of an image-visibility pair.
        
        Args:
            image: Original AIA image (H, W)
            visibilities: Original STIX visibilities (24, 2) - complex as [real, imag]
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of (augmented_image, original_visibilities) pairs
        """
        augmented_pairs = []
        
        # Always include original
        augmented_pairs.append((image.copy(), visibilities.copy()))
        
        for i in range(num_augmentations - 1):
            aug_image = image.copy()
            
            # Randomly apply augmentations
            if random.random() < self.augmentation_prob:
                
                # Gaussian blur (mild)
                if random.random() < 0.6:
                    aug_image = self.apply_gaussian_blur(aug_image)
                
                # Gaussian noise
                if random.random() < 0.7:
                    aug_image = self.apply_gaussian_noise(aug_image)
                
                # Small rotation
                if random.random() < 0.5:
                    aug_image = self.apply_rotation(aug_image)
                
                # Elastic deformation (mild)
                if random.random() < 0.4:
                    aug_image = self.apply_elastic_deformation(aug_image)
                
                # Random crop and resize (subtle zoom)
                if random.random() < 0.3:
                    aug_image = self.apply_random_crop_and_resize(aug_image)
                
                # Flips (occasionally)
                if random.random() < 0.2:
                    if random.random() < 0.5:
                        aug_image = self.apply_horizontal_flip(aug_image)
                    else:
                        aug_image = self.apply_vertical_flip(aug_image)
            
            augmented_pairs.append((aug_image, visibilities.copy()))
        
        return augmented_pairs


class AugmentedAIA2STIXDataset:
    """
    Dataset wrapper that generates augmented data on-the-fly.
    """
    
    def __init__(
        self,
        original_dataset_path: str,
        visibilities_csv: str,
        augmentor: AIA2STIXAugmentor,
        augmentations_per_sample: int = 4
    ):
        """
        Initialize augmented dataset.
        
        Args:
            original_dataset_path: Path to processed AIA images (.npy files)
            visibilities_csv: Path to visibility CSV file
            augmentor: AIA2STIXAugmentor instance
            augmentations_per_sample: Number of augmentations per original sample
        """
        self.dataset_path = Path(original_dataset_path)
        self.augmentor = augmentor
        self.augmentations_per_sample = augmentations_per_sample
        
        # Load visibility data
        self.vis_df = pd.read_csv(visibilities_csv)
        
        # Find available processed images
        self.image_files = list(self.dataset_path.glob("aia_processed_*.npy"))
        self.image_files.sort()
        
        # Create mapping from flare_id to visibility data
        self.vis_mapping = {}
        for idx, row in self.vis_df.iterrows():
            flare_id = str(row['flare_id'])
            
            # Extract visibilities (vis_01 to vis_24)
            vis_cols = [f'vis_{i:02d}' for i in range(1, 25)]
            vis_data = []
            
            for col in vis_cols:
                if col in row:
                    vis_str = row[col]
                    # Parse complex string like "(0.623-0.147j)"
                    vis_complex = complex(vis_str.strip('()'))
                    vis_data.append([vis_complex.real, vis_complex.imag])
                else:
                    vis_data.append([0.0, 0.0])
            
            self.vis_mapping[flare_id] = np.array(vis_data)  # Shape: (24, 2)
        
        print(f"Found {len(self.image_files)} processed images")
        print(f"Found {len(self.vis_mapping)} visibility entries")
    
    def extract_flare_id_from_filename(self, filename: str) -> str:
        """Extract flare_id from processed filename."""
        # Expected format: aia_processed_YYMMDDHHMMSS_rowindex.npy
        stem = Path(filename).stem
        parts = stem.split('_')
        if len(parts) >= 3:
            return parts[2]  # The YYMMDDHHMMSS part
        return ""
    
    def get_original_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get original image and visibility pair."""
        image_file = self.image_files[idx]
        flare_id = self.extract_flare_id_from_filename(image_file.name)
        
        # Load image
        image = np.load(image_file)
        
        # Get corresponding visibilities
        if flare_id in self.vis_mapping:
            visibilities = self.vis_mapping[flare_id]
        else:
            print(f"Warning: No visibility data found for flare_id {flare_id}")
            visibilities = np.zeros((24, 2))
        
        return image, visibilities, flare_id
    
    def generate_augmented_dataset(
        self, 
        output_dir: str, 
        subset_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate and save augmented dataset.
        
        Args:
            output_dir: Directory to save augmented data
            subset_size: If specified, only process first N samples
            
        Returns:
            Dictionary with generation statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        images_dir = output_path / "images"
        visibilities_dir = output_path / "visibilities" 
        images_dir.mkdir(exist_ok=True)
        visibilities_dir.mkdir(exist_ok=True)
        
        # Determine how many samples to process
        num_samples = len(self.image_files)
        if subset_size is not None:
            num_samples = min(subset_size, num_samples)
        
        augmented_metadata = []
        total_generated = 0
        
        print(f"Generating augmented dataset from {num_samples} original samples...")
        
        for i in range(num_samples):
            # Get original sample
            image, visibilities, flare_id = self.get_original_sample(i)
            
            # Generate augmentations
            augmented_pairs = self.augmentor.augment_image_visibility_pair(
                image, visibilities, self.augmentations_per_sample
            )
            
            # Save each augmented pair
            for aug_idx, (aug_image, aug_vis) in enumerate(augmented_pairs):
                sample_id = f"{flare_id}_{i:04d}_{aug_idx:02d}"
                
                # Save image
                image_path = images_dir / f"aug_image_{sample_id}.npy"
                np.save(image_path, aug_image.astype(np.float32))
                
                # Save visibilities
                vis_path = visibilities_dir / f"aug_vis_{sample_id}.npy"
                np.save(vis_path, aug_vis.astype(np.float32))
                
                # Record metadata
                augmented_metadata.append({
                    'sample_id': sample_id,
                    'original_idx': i,
                    'flare_id': flare_id,
                    'aug_idx': aug_idx,
                    'image_path': str(image_path.relative_to(output_path)),
                    'vis_path': str(vis_path.relative_to(output_path))
                })
                
                total_generated += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} original samples, "
                      f"generated {total_generated} augmented samples")
        
        # Save metadata
        metadata_df = pd.DataFrame(augmented_metadata)
        metadata_path = output_path / "augmented_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        # Save configuration
        config = {
            'augmentations_per_sample': self.augmentations_per_sample,
            'original_samples': num_samples,
            'total_augmented': total_generated,
            'augmentor_config': {
                'rotation_range': self.augmentor.rotation_range,
                'noise_std_range': self.augmentor.noise_std_range,
                'blur_sigma_range': self.augmentor.blur_sigma_range,
                'elastic_alpha_range': self.augmentor.elastic_alpha_range,
                'elastic_sigma_range': self.augmentor.elastic_sigma_range,
                'augmentation_prob': self.augmentor.augmentation_prob
            }
        }
        
        import json
        config_path = output_path / "augmentation_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\\nAugmented dataset generation complete!")
        print(f"Generated {total_generated} samples from {num_samples} originals")
        print(f"Saved to: {output_path}")
        
        return config


if __name__ == "__main__":
    # Example usage
    augmentor = AIA2STIXAugmentor(
        rotation_range=(-10, 10),  # Small rotations
        noise_std_range=(0.01, 0.06),  # Moderate noise
        blur_sigma_range=(0.2, 1.0),  # Light to moderate blur
        elastic_alpha_range=(5, 30),  # Mild elastic deformation
        elastic_sigma_range=(4, 6),
        augmentation_prob=0.8
    )
    
    # Test with sample data
    dataset = AugmentedAIA2STIXDataset(
        original_dataset_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
        visibilities_csv="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
        augmentor=augmentor,
        augmentations_per_sample=5
    )
    
    # Generate augmented dataset
    stats = dataset.generate_augmented_dataset(
        output_dir="./augmented_aia2stix_dataset",
        subset_size=50  # Test with 50 samples
    )
    
    print("\\nGeneration statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")