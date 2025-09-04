#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoder to Visibility Training Script

This script loads a pre-trained encoder (PaletteModelV2) and fine-tunes layers
to predict STIX visibilities directly from AIA images. The workflow:
1. Load pre-trained encoder checkpoint
2. Extract features at bot3 layer (48-dimensional)
3. Add new layers to predict visibilities (24x2 = 48 outputs)
4. Fine-tune the model end-to-end

@author: francesco
"""

import argparse
import os
import json
from copy import deepcopy
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm.auto import tqdm
import numpy as np

# Import modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.modules import PaletteModelV2
from src.data.dataset import get_aia2stix_data_objects


class EncoderToVisibilityModel(nn.Module):
    """
    Model that uses a pre-trained encoder and adds layers to predict visibilities.
    """
    def __init__(self, encoder_checkpoint_path, freeze_encoder=False, dropout_rate=0.1):
        super().__init__()
        
        # Load the pre-trained encoder
        self.encoder = PaletteModelV2(
            c_in=1, 
            c_out=1, 
            num_classes=None,  
            image_size=256, 
            true_img_size=256
        )
        
        # Load encoder weights
        checkpoint = torch.load(encoder_checkpoint_path, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded encoder from checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
        
        # Freeze encoder layers if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder layers frozen")
        else:
            print("Encoder layers will be fine-tuned")
        
        # Add visibility prediction layers
        # From 48-dimensional bot3 features to 48 outputs (24 visibilities x 2 components)
        self.vis_predictor = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 96),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(96, 48),  # 24 visibilities × 2 (real, imaginary)
        )
        
    def extract_encoder_features(self, x):
        """Extract 48-dimensional features from encoder at bot3 layer."""
        # Create dummy timesteps (zeros for autoencoder mode)
        batch_size = x.shape[0]
        device = x.device
        t = torch.zeros(batch_size, device=device)
        
        # Positional encoding for timesteps
        t = t.unsqueeze(-1).type(torch.float)
        t = self.encoder.pos_encoding(t, self.encoder.time_dim)
        
        # Forward pass through encoder
        x1 = self.encoder.inc(x)
        x2 = self.encoder.down1(x1, t)
        x3 = self.encoder.down2(x2, t)
        x4 = self.encoder.down3(x3, t)

        # Forward pass through bottleneck layers up to bot3
        x4 = self.encoder.bot1(x4)
        x4 = self.encoder.bot2(x4)
        bot3_features = self.encoder.bot3(x4)  # (batch_size, 48, H, W)
        
        # Global average pooling to get (batch_size, 48)
        bot3_latent = F.adaptive_avg_pool2d(bot3_features, 1)
        bot3_latent = bot3_latent.view(bot3_latent.size(0), -1)  # (batch_size, 48)
        
        return bot3_latent
    
    def forward(self, x):
        """Forward pass: extract features and predict visibilities."""
        # Extract 48-dimensional features from encoder
        features = self.extract_encoder_features(x)
        
        # Predict visibilities
        vis_pred = self.vis_predictor(features)
        
        # Reshape to (batch_size, 24, 2) for compatibility
        vis_pred = vis_pred.view(vis_pred.size(0), 24, 2)
        
        return vis_pred


def chi_square_distance(pred_vis, true_vis):
    """Calculate chi-square distance between predicted and true visibilities."""
    pred_flat = pred_vis.view(-1).cpu().numpy()
    true_flat = true_vis.view(-1).cpu().numpy()
    
    epsilon = 1e-8
    chi_sq = np.sum((pred_flat - true_flat)**2 / (np.abs(true_flat) + epsilon))
    return chi_sq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Model and data paths
    parser.add_argument('--encoder-checkpoint', type=str, required=True,
                        help='Path to the pre-trained encoder checkpoint (.pth)')
    parser.add_argument('--data-path', type=str, 
                        default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
                        help='Path to the processed AIA images')
    parser.add_argument('--vis-path', type=str,
                        default="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv", 
                        help='Path to the visibility data CSV')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of epochs to train')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder weights (only train visibility predictor)')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='Dropout rate for visibility predictor layers')
    
    # Saving and logging
    parser.add_argument('--saving-path', type=str, default="/mnt/nas05/data01/francesco/AIA2STIX/",
                        help='Path where to save the model')
    parser.add_argument('--dir-name', type=str, default='encoder_to_vis_v1',
                        help='Directory name to use for saving results')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--evaluate-every', type=int, default=5,
                        help='Evaluate model every N epochs')
    
    # Wandb logging
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--wandb-run-name', type=str,
                        help='Wandb run name')
    parser.add_argument('--wandb-project', type=str, default='aia_2_stix',
                        help='Wandb project name')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup device and directories
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    dir_path_res = os.path.join(args.saving_path, f"results_{args.dir_name}")
    dir_path_mdl = os.path.join(args.saving_path, f"model_{args.dir_name}")
    
    for dir_path in [dir_path_res, dir_path_mdl]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Initialize model
    print("Initializing model...")
    model = EncoderToVisibilityModel(
        encoder_checkpoint_path=args.encoder_checkpoint,
        freeze_encoder=args.freeze_encoder,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Model parameter analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)
    print(f'Total parameters:      {total_params:,}')
    print(f'Trainable parameters:  {trainable_params:,}')
    print(f'Frozen parameters:     {total_params - trainable_params:,}')
    print(f'Trainable percentage:  {100 * trainable_params / total_params:.2f}%')
    
    # Parameter breakdown
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    predictor_params = sum(p.numel() for p in model.vis_predictor.parameters())
    
    print(f"\\nComponent breakdown:")
    print(f'  Encoder parameters:    {encoder_params:,} ({100 * encoder_params / total_params:.2f}%)')
    print(f'  Predictor parameters:  {predictor_params:,} ({100 * predictor_params / total_params:.2f}%)')
    print("=" * 60)
    
    # Setup optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, _, train_dl = get_aia2stix_data_objects(
        vis_path=args.vis_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=args.num_workers,
        split='train',
        seed=42
    )
    
    val_dataset, _, val_dl = get_aia2stix_data_objects(
        vis_path=args.vis_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=args.num_workers,
        split='valid',
        seed=42
    )
    
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    
    # Initialize wandb if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity="francescopio",
            name=args.wandb_run_name,
            config=vars(args),
            save_code=True
        )
        wandb.watch(model)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.max_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_train_batches = len(train_dl)
        
        for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.max_epochs}")):
            # Get batch data
            aia_data = batch[0].float().to(device)  # AIA images
            vis_data = batch[1].float().to(device).reshape(-1, 24, 2)  # True visibilities
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_vis = model(aia_data)
            
            # Calculate loss
            loss = criterion(pred_vis, vis_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= num_train_batches
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = len(val_dl)
        
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validation"):
                aia_data = batch[0].float().to(device)
                vis_data = batch[1].float().to(device).reshape(-1, 24, 2)
                
                pred_vis = model(aia_data)
                loss = criterion(pred_vis, vis_data)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= num_val_batches
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        # Evaluation and visualization
        if epoch % args.evaluate_every == 0:
            # Calculate chi-square distance on a sample
            test_batch = next(iter(val_dl))
            test_aia = test_batch[0][:1].float().to(device)
            test_vis = test_batch[1][:1].float().to(device).reshape(-1, 24, 2)
            
            with torch.no_grad():
                pred_vis = model(test_aia)
                chi_sq_dist = chi_square_distance(pred_vis, test_vis)
            
            print(f"Epoch {epoch+1}: Chi-square distance = {chi_sq_dist:.6f}")
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            pred_vis_np = pred_vis[0].cpu().numpy()
            true_vis_np = test_vis[0].cpu().numpy()
            
            # Plot real and imaginary parts
            axes[0, 0].plot(pred_vis_np[:, 0], 'b-', label='Predicted Real', linewidth=2)
            axes[0, 0].plot(true_vis_np[:, 0], 'r--', label='True Real', linewidth=2)
            axes[0, 0].set_title('Real Part Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(pred_vis_np[:, 1], 'b-', label='Predicted Imag', linewidth=2)
            axes[0, 1].plot(true_vis_np[:, 1], 'r--', label='True Imag', linewidth=2)
            axes[0, 1].set_title('Imaginary Part Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot amplitude and phase
            pred_amp = np.sqrt(pred_vis_np[:, 0]**2 + pred_vis_np[:, 1]**2)
            true_amp = np.sqrt(true_vis_np[:, 0]**2 + true_vis_np[:, 1]**2)
            axes[1, 0].plot(pred_amp, 'b-', label='Predicted Amplitude', linewidth=2)
            axes[1, 0].plot(true_amp, 'r--', label='True Amplitude', linewidth=2)
            axes[1, 0].set_title('Amplitude Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            pred_phase = np.arctan2(pred_vis_np[:, 1], pred_vis_np[:, 0])
            true_phase = np.arctan2(true_vis_np[:, 1], true_vis_np[:, 0])
            axes[1, 1].plot(pred_phase, 'b-', label='Predicted Phase', linewidth=2)
            axes[1, 1].plot(true_phase, 'r--', label='True Phase', linewidth=2)
            axes[1, 1].set_title('Phase Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            fig.suptitle(f'Encoder→Vis - Epoch {epoch+1} - Chi²: {chi_sq_dist:.6f}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(dir_path_res, f'visibility_comparison_epoch_{epoch+1}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'chi_square_distance': chi_sq_dist,
                    'visibility_comparison': wandb.Image(fig)
                })
            
            plt.close(fig)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch_val_loss < best_val_loss:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                is_best = True
            else:
                is_best = False
                
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }
            
            # Save regular checkpoint
            ckpt_path = os.path.join(dir_path_mdl, f'encoder_to_vis_epoch_{epoch+1:04d}.pth')
            torch.save(checkpoint, ckpt_path)
            
            # Save best model
            if is_best:
                best_path = os.path.join(dir_path_mdl, 'encoder_to_vis_best.pth')
                torch.save(checkpoint, best_path)
                print(f"New best model saved with val_loss: {epoch_val_loss:.6f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()