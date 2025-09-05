#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Augmented AIA2STIX Dataset

Script to generate augmented training data for the AIA2STIX project.
Applies spatial transformations and noise while preserving visibility relationships.

Usage:
    python generate_augmented_dataset.py --subset-size 100 --aug-per-sample 6
    python generate_augmented_dataset.py --full-dataset --output-dir ./full_augmented_data

@author: francesco
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data_augmentation import AIA2STIXAugmentor, AugmentedAIA2STIXDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Data paths
    parser.add_argument('--data-path', type=str, 
                        default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
                        help='Path to the processed AIA images')
    parser.add_argument('--vis-path', type=str,
                        default="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv", 
                        help='Path to the visibility data CSV')
    parser.add_argument('--output-dir', type=str, default="./augmented_aia2stix_dataset",
                        help='Output directory for augmented dataset')
    
    # Dataset generation parameters
    parser.add_argument('--subset-size', type=int, default=None,
                        help='Process only first N samples (for testing)')
    parser.add_argument('--full-dataset', action='store_true',
                        help='Process the full dataset')
    parser.add_argument('--aug-per-sample', type=int, default=5,
                        help='Number of augmented versions per original sample')
    
    # Augmentation parameters
    parser.add_argument('--rotation-range', type=float, nargs=2, default=[-12, 12],
                        help='Rotation angle range in degrees')
    parser.add_argument('--noise-std-range', type=float, nargs=2, default=[0.01, 0.07],
                        help='Gaussian noise standard deviation range')
    parser.add_argument('--blur-sigma-range', type=float, nargs=2, default=[0.2, 1.2],
                        help='Gaussian blur sigma range')
    parser.add_argument('--elastic-alpha-range', type=float, nargs=2, default=[5, 40],
                        help='Elastic deformation strength range')
    parser.add_argument('--elastic-sigma-range', type=float, nargs=2, default=[4, 7],
                        help='Elastic deformation smoothness range')
    parser.add_argument('--augmentation-prob', type=float, default=0.85,
                        help='Probability of applying augmentation')
    
    args = parser.parse_args()
    
    # Determine subset size
    subset_size = None if args.full_dataset else args.subset_size
    
    print("=" * 60)
    print("AIA2STIX AUGMENTED DATASET GENERATOR")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Visibilities CSV: {args.vis_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing: {'Full dataset' if subset_size is None else f'First {subset_size} samples'}")
    print(f"Augmentations per sample: {args.aug_per_sample}")
    print()
    
    # Initialize augmentor
    print("Initializing augmentor with parameters:")
    print(f"  Rotation range: {args.rotation_range} degrees")
    print(f"  Noise std range: {args.noise_std_range}")
    print(f"  Blur sigma range: {args.blur_sigma_range}")
    print(f"  Elastic alpha range: {args.elastic_alpha_range}")
    print(f"  Elastic sigma range: {args.elastic_sigma_range}")
    print(f"  Augmentation probability: {args.augmentation_prob}")
    print()
    
    augmentor = AIA2STIXAugmentor(
        rotation_range=tuple(args.rotation_range),
        noise_std_range=tuple(args.noise_std_range),
        blur_sigma_range=tuple(args.blur_sigma_range),
        elastic_alpha_range=tuple(args.elastic_alpha_range),
        elastic_sigma_range=tuple(args.elastic_sigma_range),
        augmentation_prob=args.augmentation_prob
    )
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = AugmentedAIA2STIXDataset(
        original_dataset_path=args.data_path,
        visibilities_csv=args.vis_path,
        augmentor=augmentor,
        augmentations_per_sample=args.aug_per_sample
    )
    print()
    
    # Generate augmented dataset
    print("Starting dataset generation...")
    try:
        stats = dataset.generate_augmented_dataset(
            output_dir=args.output_dir,
            subset_size=subset_size
        )
        
        print("\\n" + "=" * 60)
        print("GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Original samples processed: {stats['original_samples']}")
        print(f"Total augmented samples: {stats['total_augmented']}")
        print(f"Augmentations per original: {stats['augmentations_per_sample']}")
        print(f"Dataset expansion factor: {stats['total_augmented'] / stats['original_samples']:.1f}x")
        print(f"Output directory: {args.output_dir}")
        
        # Calculate approximate disk usage
        approx_mb_per_sample = 1.0  # Estimate: 512x512 float32 ≈ 1MB
        total_mb = stats['total_augmented'] * approx_mb_per_sample
        print(f"Estimated disk usage: ~{total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
        print()
        
        print("Files created:")
        print(f"  - {args.output_dir}/images/ (augmented images)")
        print(f"  - {args.output_dir}/visibilities/ (corresponding visibilities)")
        print(f"  - {args.output_dir}/augmented_metadata.csv (sample metadata)")
        print(f"  - {args.output_dir}/augmentation_config.json (configuration)")
        
    except Exception as e:
        print(f"\\nERROR during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)