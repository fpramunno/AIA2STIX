#!/usr/bin/env python3
"""
Generate visibilities from AIA images using trained diffusion models.

This script loads trained diffusion models and generates visibility data 
for training the refiner model.
"""

import argparse
import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import safetensors.torch as safetorch

import src as K
from src.data.dataset import get_aia2stix_data_objects
from util import generate_samples

class VisibilityGenerator:
    def __init__(self, config_path, model_path, device='cuda'):
        """
        Initialize the visibility generator.
        
        Args:
            config_path: Path to model configuration file
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.config = K.config.load_config(config_path)
        
        # Load model
        inner_model = K.config.make_model(self.config)
        self.inner_model_ema = deepcopy(inner_model)
        self.model_ema = K.config.make_denoiser_wrapper(self.config)(self.inner_model_ema)
        
        # Load checkpoint
        print(f"Loading model from: {model_path}")
        if model_path.endswith('.safetensors'):
            state_dict = safetorch.load_file(model_path)
            # Look for the appropriate key in the state dict
            if 'model_ema' in state_dict:
                model_state = {k.replace('model_ema.', ''): v for k, v in state_dict.items() if k.startswith('model_ema.')}
            elif 'inner_model' in str(list(state_dict.keys())[0]):
                model_state = {k.replace('inner_model.', ''): v for k, v in state_dict.items() if 'inner_model' in k}
            else:
                model_state = state_dict
        else:
            ckpt = torch.load(model_path, map_location='cpu')
            if 'model_ema' in ckpt:
                model_state = ckpt['model_ema']
            elif 'inner_model' in ckpt:
                model_state = ckpt['inner_model']
            else:
                model_state = ckpt
        
        self.model_ema.inner_model.load_state_dict(model_state)
        self.model_ema.to(self.device)
        self.model_ema.eval()
        
        print("✅ Model loaded successfully!")
    
    def generate_batch_visibilities(self, aia_batch, sampler="dpmpp_2m", num_steps=50, num_samples_per_image=1):
        """
        Generate visibilities from a batch of AIA images.
        
        Args:
            aia_batch: Batch of encoded AIA images from dataset (should match training format)
            sampler: Sampler to use for generation
            num_steps: Number of diffusion steps
            num_samples_per_image: Number of visibility samples per AIA image
            
        Returns:
            Generated visibilities (B*num_samples_per_image, 24, 2)
        """
        all_visibilities = []
        
        with torch.no_grad():
            for i, aia_encoded in enumerate(aia_batch):
                aia_encoded = aia_encoded.to(self.device)
                
                # Based on the training code, enc_inpt should be reshaped to (batch, 1, 24, 2)
                # The training code does: enc_inpt = batch[2].to(device).reshape(inpt.shape[0], 1, 24, 2)
                
                if aia_encoded.dim() == 1:
                    # If 1D, assume it's already flattened (24*2=48 elements)
                    if len(aia_encoded) == 48:
                        # Reshape from (48,) to (1, 1, 24, 2)
                        cond_tensor = aia_encoded.reshape(1, 1, 24, 2)
                    else:
                        print(f"Warning: Expected 48 elements for encoded AIA, got {len(aia_encoded)}")
                        # Pad or truncate to 48 elements
                        if len(aia_encoded) < 48:
                            padded = torch.zeros(48, device=self.device)
                            padded[:len(aia_encoded)] = aia_encoded
                        else:
                            padded = aia_encoded[:48]
                        cond_tensor = padded.reshape(1, 1, 24, 2)
                        
                elif aia_encoded.dim() == 3 and aia_encoded.shape == (1, 24, 2):
                    # Already in the right shape, just ensure batch dimension
                    cond_tensor = aia_encoded.unsqueeze(0) if aia_encoded.shape[0] == 1 else aia_encoded
                    
                elif aia_encoded.dim() == 2:
                    if aia_encoded.shape == (24, 2):
                        # Add batch dimension: (24, 2) -> (1, 1, 24, 2)
                        cond_tensor = aia_encoded.unsqueeze(0).unsqueeze(0)
                    elif aia_encoded.shape[1] == 48:
                        # Assume (batch, 48) and reshape
                        batch_size = aia_encoded.shape[0]
                        cond_tensor = aia_encoded.reshape(batch_size, 1, 24, 2)
                    else:
                        # Try to flatten and reshape
                        flattened = aia_encoded.flatten()
                        if len(flattened) >= 48:
                            cond_tensor = flattened[:48].reshape(1, 1, 24, 2)
                        else:
                            padded = torch.zeros(48, device=self.device)
                            padded[:len(flattened)] = flattened
                            cond_tensor = padded.reshape(1, 1, 24, 2)
                else:
                    # For any other shape, try to flatten and reshape
                    flattened = aia_encoded.flatten()
                    if len(flattened) >= 48:
                        cond_tensor = flattened[:48].reshape(1, 1, 24, 2)
                    else:
                        padded = torch.zeros(48, device=self.device)
                        padded[:len(flattened)] = flattened
                        cond_tensor = padded.reshape(1, 1, 24, 2)
                
                print(f"Processing AIA encoded shape: {aia_encoded.shape} -> cond shape: {cond_tensor.shape}")
                
                # Generate multiple samples for this image if requested
                for _ in range(num_samples_per_image):
                    visibilities = generate_samples(
                        model=self.model_ema,
                        num_samples=1,
                        device=self.device,
                        cond_label=None,
                        sampler=sampler,
                        step=num_steps,
                        cond_img=cond_tensor
                    )
                    all_visibilities.append(visibilities.squeeze(0))  # Remove batch dim
        
        return torch.stack(all_visibilities)
    
    def process_dataset(self, data_loader, output_dir, sampler="dpmpp_2m", 
                       num_steps=50, num_samples_per_image=1, max_batches=None):
        """
        Process entire dataset and save generated visibilities.
        
        Args:
            data_loader: DataLoader for AIA images
            output_dir: Directory to save outputs
            sampler: Sampler to use
            num_steps: Number of diffusion steps
            num_samples_per_image: Number of samples per image
            max_batches: Maximum number of batches to process (None for all)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_generated_vis = []
        all_true_vis = []
        all_aia_images = []
        
        print(f"Processing dataset with sampler: {sampler}, steps: {num_steps}")
        
        processed_batches = 0
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            if max_batches and processed_batches >= max_batches:
                break
                
            aia_images, true_visibilities, enc_aia = batch
            
            # Generate visibilities from AIA images
            generated_vis = self.generate_batch_visibilities(
                enc_aia,
                sampler=sampler,
                num_steps=num_steps,
                num_samples_per_image=num_samples_per_image
            )
            
            # Store results
            all_generated_vis.append(generated_vis.cpu())
            
            # Handle true visibilities (repeat if we generate multiple samples per image)
            if num_samples_per_image > 1:
                true_vis_repeated = true_visibilities.repeat_interleave(num_samples_per_image, dim=0)
                aia_repeated = aia_images.repeat_interleave(num_samples_per_image, dim=0)
            else:
                true_vis_repeated = true_visibilities
                aia_repeated = aia_images
            
            all_true_vis.append(true_vis_repeated)
            all_aia_images.append(aia_repeated)
            
            processed_batches += 1
            
            # Save intermediate results every 100 batches
            # if (batch_idx + 1) % 100 == 0:
            #     self._save_intermediate_results(output_dir, all_generated_vis, all_true_vis, 
            #                                    all_aia_images, batch_idx + 1)
        
        # Final save
        print("Saving final results...")
        generated_vis_tensor = torch.cat(all_generated_vis, dim=0)
        true_vis_tensor = torch.cat(all_true_vis, dim=0)
        aia_tensor = torch.cat(all_aia_images, dim=0)
        
        # Save as numpy arrays
        np.save(output_dir / "generated_visibilities.npy", generated_vis_tensor.numpy())
        np.save(output_dir / "true_visibilities.npy", true_vis_tensor.numpy())
        # np.save(output_dir / "aia_images.npy", aia_tensor.numpy())
        
        # Save metadata
        metadata = {
            "num_samples": len(generated_vis_tensor),
            "sampler": sampler,
            "num_steps": num_steps,
            "num_samples_per_image": num_samples_per_image,
            "visibility_shape": list(generated_vis_tensor.shape[1:]),
            "aia_shape": list(aia_tensor.shape[1:])
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved {len(generated_vis_tensor)} visibility samples to {output_dir}")
        print(f"Generated visibilities shape: {generated_vis_tensor.shape}")
        print(f"True visibilities shape: {true_vis_tensor.shape}")
        
        return generated_vis_tensor, true_vis_tensor, aia_tensor
    
    def _save_intermediate_results(self, output_dir, generated_vis, true_vis, aia_images, batch_num):
        """Save intermediate results."""
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        
        generated_vis_tensor = torch.cat(generated_vis, dim=0)
        true_vis_tensor = torch.cat(true_vis, dim=0)
        aia_tensor = torch.cat(aia_images, dim=0)
        
        np.save(intermediate_dir / f"generated_vis_batch_{batch_num}.npy", generated_vis_tensor.numpy())
        np.save(intermediate_dir / f"true_vis_batch_{batch_num}.npy", true_vis_tensor.numpy())
        np.save(intermediate_dir / f"aia_batch_{batch_num}.npy", aia_tensor.numpy())
        
        print(f"💾 Saved intermediate results up to batch {batch_num}")

def create_data_loader(data_path, batch_size=8, num_workers=4, split='train'):
    """Create data loader for AIA images."""
    train_dataset, train_sampler, train_dl = get_aia2stix_data_objects(
        data_path=data_path,
        vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
        split=split,
        enc_data_path="/mnt/nas05/astrodata01/aia_2_stix/encoded_data/",
        seed=42,
        distributed=False,
        batch_size=batch_size,
        num_data_workers=num_workers
    )
    
    data_loader = train_dl
    
    return data_loader

def main():
    parser = argparse.ArgumentParser(description='Generate visibilities from AIA using trained diffusion models')
    
    # Model and config paths
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration file (.json)')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Data paths
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to AIA dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save generated visibilities')
    
    # Generation parameters
    parser.add_argument('--sampler', type=str, default='dpmpp_2m',
                       choices=['euler', 'euler_ancestral', 'heun', 'dpm_2', 
                               'dpm_2_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde'],
                       help='Diffusion sampler to use')
    parser.add_argument('--num-steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--num-samples-per-image', type=int, default=1,
                       help='Number of visibility samples to generate per AIA image')
    
    # Processing parameters
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to process (for testing)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to use')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generation parameters
    with open(output_dir / "generation_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize generator
    print("Initializing visibility generator...")
    generator = VisibilityGenerator(
        config_path=args.config,
        model_path=args.model_path,
        device=args.device
    )
    
    # Create data loader
    print(f"Loading dataset from: {args.data_path}")
    data_loader = create_data_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split
    )
    
    print(f"Dataset size: {len(data_loader.dataset)} samples")
    print(f"Number of batches: {len(data_loader)}")
    
    # Generate visibilities
    generated_vis, true_vis, aia_images = generator.process_dataset(
        data_loader=data_loader,
        output_dir=args.output_dir,
        sampler=args.sampler,
        num_steps=args.num_steps,
        num_samples_per_image=args.num_samples_per_image,
        max_batches=args.max_batches
    )
    
    print("🎉 Visibility generation completed!")
    print(f"Ready for refiner training with:")
    print(f"  Generated visibilities: {output_dir}/generated_visibilities.npy")
    print(f"  True visibilities: {output_dir}/true_visibilities.npy")

if __name__ == '__main__':
    main()