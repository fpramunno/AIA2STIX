#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model encoding script for extracting latent representations at bot3 layer
Loads a trained model checkpoint and encodes data to 48-dimensional latent space
Saves encoded data to separate train and test directories

@author: francesco
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Import modules from training directory
import sys
sys.path.append(str(Path(__file__).parent / "training"))

from src.modules import PaletteModelV2
from src.data.dataset import get_aia2stix_data_objects


def extract_bot3_features(model, x, y, t):
    """
    Extract features at the bot3 layer (48-dimensional latent space).
    This requires a custom forward pass through the model up to bot3.
    """
    model.eval()
    
    with torch.no_grad():
        # Positional encoding for timesteps
        t = t.unsqueeze(-1).type(torch.float)
        t = model.pos_encoding(t, model.time_dim)
        
        # Concatenate input and reference if provided
        if y is not None:
            x = torch.cat([x, y], dim=1)
        
        # Forward pass through encoder
        x1 = model.inc(x)
        x2 = model.down1(x1, t)
        x3 = model.down2(x2, t)
        x4 = model.down3(x3, t)

        # Forward pass through bottleneck layers up to bot3
        x4 = model.bot1(x4)
        x4 = model.bot2(x4)
        bot3_features = model.bot3(x4)  # This should be (batch_size, 48, H, W)
        
        # Apply global average pooling to get (batch_size, 48)
        bot3_latent = torch.nn.functional.adaptive_avg_pool2d(bot3_features, 1)
        bot3_latent = bot3_latent.view(bot3_latent.size(0), -1)  # (batch_size, 48)
        
    return bot3_latent


def encode_dataset(model, dataloader, device, split_name, output_dir):
    """Encode a dataset split and save results."""
    
    # Create output directory for this split
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    encoded_features = []
    filenames = []
    
    model.eval()
    
    print(f"Encoding {split_name} dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Encoding {split_name}")):
            # Handle batch structure
            if isinstance(batch, dict):
                aia_data = batch[0].to(device)
                vis_data = batch[1].to(device)
            else:
                aia_data, vis_data = batch
                aia_data = aia_data.to(device)
                vis_data = vis_data.to(device)
            
            # Create dummy timesteps (all zeros for autoencoder mode)
            batch_size = aia_data.shape[0]
            t = torch.zeros(batch_size, device=device)
            
            # Extract bot3 features (48-dimensional latent space)
            bot3_latent = extract_bot3_features(model, aia_data, y=None, t=t)
            
            # Move to CPU and store
            bot3_latent_cpu = bot3_latent.cpu().numpy()
            encoded_features.append(bot3_latent_cpu)
            
            # Generate filenames for this batch
            for i in range(batch_size):
                filename = f"{split_name}_batch_{batch_idx:04d}_sample_{i:02d}.npy"
                filenames.append(filename)
    
    # Concatenate all features
    all_features = np.concatenate(encoded_features, axis=0)
    print(f"Encoded {len(all_features)} samples from {split_name} split")
    print(f"Feature shape: {all_features.shape}")
    
    # Save individual files
    for i, (features, filename) in enumerate(zip(all_features, filenames)):
        file_path = split_dir / filename
        np.save(file_path, features)
    
    
    print(f"Saved {len(filenames)} individual files and concatenated file to {split_dir}")
    
    return all_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--data-path', type=str,
                        default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
                        help='Path to AIA processed images OR augmented dataset directory (use with --use-augmented)')
    parser.add_argument('--vis-path', type=str,
                        default="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
                        help='Path to the visibility data CSV (ignored if --use-augmented is set)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save encoded features')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--encode-splits', nargs='+', default=['train', 'valid'],
                        choices=['train', 'valid', 'test'],
                        help='Which data splits to encode')
    parser.add_argument('--use-augmented', action='store_true',
                        help='Use augmented dataset format (data-path should point to augmented dataset directory)')
    parser.add_argument('--date-ranges', type=str, nargs='*',
                        help='Date ranges in format: train 210520 210630 valid 210701 210730 (for augmented datasets)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse date ranges if provided
    date_ranges = None
    if args.date_ranges and len(args.date_ranges) >= 3:
        date_ranges = {}
        i = 0
        while i < len(args.date_ranges) - 2:
            split_name = args.date_ranges[i]
            start_date = args.date_ranges[i + 1]
            end_date = args.date_ranges[i + 2]
            date_ranges[split_name] = (start_date, end_date)
            i += 3
        print(f"Using date ranges: {date_ranges}")
    
    # Print dataset configuration
    if args.use_augmented:
        print(f"Using AUGMENTED dataset from: {args.data_path}")
        print("Note: --vis-path will be ignored for augmented datasets")
    else:
        print(f"Using ORIGINAL dataset from: {args.data_path}")
        print(f"Visibility data from: {args.vis_path}")
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}")
    
    # Initialize model with same configuration as training
    model = PaletteModelV2(
        c_in=1, 
        c_out=1, 
        num_classes=None,  
        image_size=256, 
        true_img_size=256
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # Encode specified splits
    for split in args.encode_splits:
        print(f"\n{'='*50}")
        print(f"Processing {split} split")
        print(f"{'='*50}")

        if split == 'train':
            output_dir = Path(args.output_dir) / 'train'
        elif split == 'valid':
            output_dir = Path(args.output_dir) / 'valid'
        elif split == 'test':
            output_dir = Path(args.output_dir) / 'test'

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        dataset, _, dataloader = get_aia2stix_data_objects(
            vis_path=None if args.use_augmented else args.vis_path,
            data_path=args.data_path,
            batch_size=args.batch_size,
            distributed=False,
            num_data_workers=args.num_workers,
            split=split,
            seed=42,
            use_augmented=args.use_augmented,
            date_ranges=date_ranges
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Number of batches: {len(dataloader)}")
        
        # Encode the dataset
        encoded_features = encode_dataset(
            model=model,
            dataloader=dataloader,
            device=device,
            split_name=split,
            output_dir=output_dir
        )
        
        print(f"Completed encoding {split} split!")
        print(f"Encoded features shape: {encoded_features.shape}")
    
    print(f"\n{'='*50}")
    print("Encoding completed successfully!")
    print(f"All encoded features saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()