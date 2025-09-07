#!/usr/bin/env python3

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import io

from src.modules import VisibilityRefiner, ChiSquareLoss, CompositeLoss

class VisibilityDataset(Dataset):
    def __init__(self, diffusion_visibilities, true_visibilities):
        """
        Dataset for training the visibility refiner.
        
        Args:
            diffusion_visibilities: Generated visibilities from diffusion model (N, 1, 24, 2) or (N, 24, 2)
            true_visibilities: Ground truth visibilities (N, 24, 2)
        """
        print(f"Original diffusion visibilities shape: {diffusion_visibilities.shape}")
        print(f"Original true visibilities shape: {true_visibilities.shape}")
        
        # Handle the case where diffusion visibilities have an extra dimension (N, 1, 24, 2)
        if len(diffusion_visibilities.shape) == 4 and diffusion_visibilities.shape[1] == 1:
            # Squeeze out the singleton dimension: (N, 1, 24, 2) -> (N, 24, 2)
            diffusion_visibilities = diffusion_visibilities.squeeze(1)
            print(f"Squeezed diffusion visibilities shape: {diffusion_visibilities.shape}")
        
        # Ensure both have the same shape now
        assert diffusion_visibilities.shape == true_visibilities.shape, \
            f"Shape mismatch after processing: diffusion {diffusion_visibilities.shape} vs true {true_visibilities.shape}"
        
        # Flatten (24, 2) -> (48) for processing
        self.diffusion_vis = diffusion_visibilities.reshape(-1, 48)
        self.true_vis = true_visibilities.reshape(-1, 48)
        
        print(f"Final dataset size: {len(self.diffusion_vis)} samples with 48 features each")
        
    def __len__(self):
        return len(self.diffusion_vis)
    
    def __getitem__(self, idx):
        return {
            'diffusion_vis': torch.FloatTensor(self.diffusion_vis[idx]),
            'true_vis': torch.FloatTensor(self.true_vis[idx])
        }

class RefinerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize model
        self.model = VisibilityRefiner(
            input_dim=config['model']['input_dim'],
            hidden_dims=config['model']['hidden_dims'],
            output_dim=config['model']['output_dim'],
            dropout_rate=config['model']['dropout_rate'],
            use_residual=config['model']['use_residual']
        ).to(self.device)
        
        # Print model summary
        self._print_model_summary()
        
        # Loss function
        if config['loss']['type'] == 'chi_square':
            self.criterion = ChiSquareLoss(epsilon=float(config['loss']['epsilon']))
        elif config['loss']['type'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['loss']['type'] == 'composite':
            self.criterion = CompositeLoss(
                primary_loss_type=config['loss']['primary_type'],
                primary_weight=float(config['loss']['primary_weight']),
                real_imag_weight=float(config['loss']['real_imag_weight']),
                epsilon=float(config['loss']['epsilon'])
            )
        else:
            raise ValueError(f"Unsupported loss type: {config['loss']['type']}")
        
        # Optimizer - ensure all parameters are float type
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['optimizer']['lr']),
            weight_decay=float(config['optimizer']['weight_decay'])
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=float(config['scheduler']['factor']),
            patience=int(config['scheduler']['patience'])
        )
        
        self.best_loss = float('inf')
        self.best_chi_square = float('inf')
    
    def _print_model_summary(self):
        """Print detailed model parameter analysis."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print("=" * 60)
        print("REFINER MODEL PARAMETER SUMMARY")
        print("=" * 60)
        print(f'Total parameters:      {total_params:,}')
        print(f'Trainable parameters:  {trainable_params:,}')
        print(f'Non-trainable params:  {non_trainable_params:,}')
        print(f'Trainable percentage:  {100 * trainable_params / total_params:.2f}%')
        
        # Memory estimation (assuming float32)
        param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
        print(f'Model size (approx):   {param_size_mb:.2f} MB')
        
        # Model architecture summary
        print(f'\nModel Architecture:')
        print(f'  Input dimension:     {self.config["model"]["input_dim"]}')
        print(f'  Hidden dimensions:   {self.config["model"]["hidden_dims"]}')
        print(f'  Output dimension:    {self.config["model"]["output_dim"]}')
        print(f'  Dropout rate:        {self.config["model"]["dropout_rate"]}')
        print(f'  Residual connections: {self.config["model"]["use_residual"]}')
        
        # Parameter breakdown by layer type
        param_breakdown = {}
        for name, param in self.model.named_parameters():
            # Extract layer type (e.g., 'network.0' -> 'Linear', 'network.1' -> 'BatchNorm1d')
            if 'network' in name:
                parts = name.split('.')
                if len(parts) >= 2:
                    layer_idx = int(parts[1])
                    # Get the actual layer to determine type
                    layer = self.model.network[layer_idx]
                    layer_type = layer.__class__.__name__
                else:
                    layer_type = 'network'
            else:
                layer_type = name.split('.')[0] if '.' in name else name
                
            if layer_type not in param_breakdown:
                param_breakdown[layer_type] = 0
            param_breakdown[layer_type] += param.numel()
        
        print("\nParameter breakdown by layer type:")
        for layer_type, count in sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / total_params
            print(f'  {layer_type:20}: {count:>10,} ({percentage:5.2f}%)')
        
        print("=" * 60)
        
    def chi_square_distance(self, pred_vis, true_vis):
        """Calculate chi-square distance between predicted and true visibilities."""
        # pred_vis and true_vis are already flattened (batch, 48)
        pred_flat = pred_vis.view(-1).cpu().numpy()
        true_flat = true_vis.view(-1).cpu().numpy()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        chi_sq = np.sum((pred_flat - true_flat)**2 / (np.abs(true_flat) + epsilon))
        return chi_sq
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            diffusion_vis = batch['diffusion_vis'].to(self.device)
            true_vis = batch['true_vis'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            refined_vis = self.model(diffusion_vis)
            loss = self.criterion(refined_vis, true_vis)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                diffusion_vis = batch['diffusion_vis'].to(self.device)
                true_vis = batch['true_vis'].to(self.device)
                
                refined_vis = self.model(diffusion_vis)
                loss = self.criterion(refined_vis, true_vis)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def create_visualization(self, diffusion_vis, refined_vis, true_vis, epoch, chi_sq_before, chi_sq_after):
        """Create visualization plot comparing diffusion, refined, and true visibilities."""
        # Reshape from (48,) to (24, 2) for plotting
        diffusion_vis_reshaped = diffusion_vis.reshape(24, 2)
        refined_vis_reshaped = refined_vis.reshape(24, 2)
        true_vis_reshaped = true_vis.reshape(24, 2)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot real parts
        axes[0, 0].plot(diffusion_vis_reshaped[:, 0], 'g:', label='Diffusion Real', linewidth=2, alpha=0.7)
        axes[0, 0].plot(refined_vis_reshaped[:, 0], 'b-', label='Refined Real', linewidth=2)
        axes[0, 0].plot(true_vis_reshaped[:, 0], 'r--', label='True Real', linewidth=2)
        axes[0, 0].set_title('Real Part Comparison')
        axes[0, 0].set_xlabel('Visibility Index')
        axes[0, 0].set_ylabel('Real Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot imaginary parts
        axes[0, 1].plot(diffusion_vis_reshaped[:, 1], 'g:', label='Diffusion Imag', linewidth=2, alpha=0.7)
        axes[0, 1].plot(refined_vis_reshaped[:, 1], 'b-', label='Refined Imag', linewidth=2)
        axes[0, 1].plot(true_vis_reshaped[:, 1], 'r--', label='True Imag', linewidth=2)
        axes[0, 1].set_title('Imaginary Part Comparison')
        axes[0, 1].set_xlabel('Visibility Index')
        axes[0, 1].set_ylabel('Imaginary Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot amplitude comparison
        diffusion_amp = np.sqrt(diffusion_vis_reshaped[:, 0]**2 + diffusion_vis_reshaped[:, 1]**2)
        refined_amp = np.sqrt(refined_vis_reshaped[:, 0]**2 + refined_vis_reshaped[:, 1]**2)
        true_amp = np.sqrt(true_vis_reshaped[:, 0]**2 + true_vis_reshaped[:, 1]**2)
        
        axes[1, 0].plot(diffusion_amp, 'g:', label='Diffusion Amplitude', linewidth=2, alpha=0.7)
        axes[1, 0].plot(refined_amp, 'b-', label='Refined Amplitude', linewidth=2)
        axes[1, 0].plot(true_amp, 'r--', label='True Amplitude', linewidth=2)
        axes[1, 0].set_title('Amplitude Comparison')
        axes[1, 0].set_xlabel('Visibility Index')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot phase comparison
        diffusion_phase = np.arctan2(diffusion_vis_reshaped[:, 1], diffusion_vis_reshaped[:, 0])
        refined_phase = np.arctan2(refined_vis_reshaped[:, 1], refined_vis_reshaped[:, 0])
        true_phase = np.arctan2(true_vis_reshaped[:, 1], true_vis_reshaped[:, 0])
        
        axes[1, 1].plot(diffusion_phase, 'g:', label='Diffusion Phase', linewidth=2, alpha=0.7)
        axes[1, 1].plot(refined_phase, 'b-', label='Refined Phase', linewidth=2)
        axes[1, 1].plot(true_phase, 'r--', label='True Phase', linewidth=2)
        axes[1, 1].set_title('Phase Comparison')
        axes[1, 1].set_xlabel('Visibility Index')
        axes[1, 1].set_ylabel('Phase (radians)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add overall title with chi-square distances
        improvement = ((chi_sq_before - chi_sq_after) / chi_sq_before * 100)
        fig.suptitle(f'Refiner Results - Epoch {epoch}\n'
                    f'Chi-square Before: {chi_sq_before:.6f} → After: {chi_sq_after:.6f} '
                    f'(Improvement: {improvement:.2f}%)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def evaluate_sample(self, val_loader, epoch):
        """Evaluate refiner on a sample and create visualization."""
        self.model.eval()
        
        # Get a single batch for evaluation
        sample_batch = next(iter(val_loader))
        diffusion_vis = sample_batch['diffusion_vis'][:1].to(self.device)  # Take first sample
        true_vis = sample_batch['true_vis'][:1].to(self.device)
        
        with torch.no_grad():
            refined_vis = self.model(diffusion_vis)
            
            # Calculate chi-square distances
            chi_sq_before = self.chi_square_distance(diffusion_vis, true_vis)
            chi_sq_after = self.chi_square_distance(refined_vis, true_vis)
            
            # Convert to numpy for plotting
            diffusion_np = diffusion_vis[0].cpu().numpy()
            refined_np = refined_vis[0].cpu().numpy()
            true_np = true_vis[0].cpu().numpy()
            
            # Create visualization
            fig = self.create_visualization(diffusion_np, refined_np, true_np, 
                                          epoch, chi_sq_before, chi_sq_after)
            
            return fig, chi_sq_before, chi_sq_after
    
    def train(self, train_loader, val_loader=None):
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        # Create results directory for plots
        results_dir = Path(self.config['checkpoint_dir']) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader)
            
            log_dict = {'epoch': epoch, 'train_loss': train_loss, 'lr': self.optimizer.param_groups[0]['lr']}
            
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.scheduler.step(val_loss)
                log_dict['val_loss'] = val_loss
                
                # Evaluation and visualization every 10 epochs or at the end
                if epoch % 10 == 0 or epoch == self.config['epochs'] - 1:
                    fig, chi_sq_before, chi_sq_after = self.evaluate_sample(val_loader, epoch)
                    
                    # Save plot
                    plot_path = results_dir / f'refiner_comparison_epoch_{epoch:04d}.png'
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    print(f"Visualization saved to: {plot_path}")
                    
                    # Add to log
                    log_dict['chi_square_before'] = chi_sq_before
                    log_dict['chi_square_after'] = chi_sq_after
                    log_dict['chi_square_improvement'] = (chi_sq_before - chi_sq_after) / chi_sq_before * 100
                    
                    # Log to wandb if enabled
                    if self.config.get('use_wandb', False):
                        log_dict['refiner_visualization'] = wandb.Image(fig)
                    
                    plt.close(fig)
                    
                    print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                          f"Chi-square: {chi_sq_before:.6f} → {chi_sq_after:.6f} "
                          f"({((chi_sq_before - chi_sq_after) / chi_sq_before * 100):+.2f}%)")
                else:
                    print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Save best model based on validation loss
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, 'best_model.pth')
                    
            else:
                print(f"Epoch {epoch+1}/{self.config['epochs']} - Train Loss: {train_loss:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log(log_dict)
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, epoch, filename):
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"Checkpoint saved: {checkpoint_dir / filename}")

def load_data(diffusion_path, true_path):
    """Load diffusion model outputs and true visibilities."""
    diffusion_vis = np.load(diffusion_path)  # Shape: (N, 24, 2)
    true_vis = np.load(true_path)  # Shape: (N, 24, 2)
    
    print(f"Loaded diffusion visibilities: {diffusion_vis.shape}")
    print(f"Loaded true visibilities: {true_vis.shape}")
    
    return diffusion_vis, true_vis

def create_data_loaders(diffusion_vis, true_vis, config):
    """Create train/validation data loaders."""
    dataset = VisibilityDataset(diffusion_vis, true_vis)
    
    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    ) if val_size > 0 else None
    
    return train_loader, val_loader

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train Visibility Refiner')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    # Training data
    parser.add_argument('--train-diffusion-data', type=str, required=True, 
                       help='Path to training diffusion model visibility outputs (.npy)')
    parser.add_argument('--train-true-data', type=str, required=True,
                       help='Path to training true visibilities (.npy)')
    
    # Validation data (optional - if not provided, will split training data)
    parser.add_argument('--val-diffusion-data', type=str, default=None,
                       help='Path to validation diffusion model visibility outputs (.npy)')
    parser.add_argument('--val-true-data', type=str, default=None,
                       help='Path to validation true visibilities (.npy)')
    
    # Legacy support for single dataset
    parser.add_argument('--diffusion-data', type=str, default=None,
                       help='[DEPRECATED] Path to diffusion model visibility outputs (.npy)')
    parser.add_argument('--true-data', type=str, default=None,
                       help='[DEPRECATED] Path to true visibilities (.npy)')
    
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # Training parameter overrides
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config file)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config file)')
    
    # Weights & Biases arguments
    parser.add_argument('--use-wandb', type=str, default='false', 
                       help='Enable Weights & Biases logging (true/false)')
    parser.add_argument('--wandb-project', type=str, default='aia2stix_refiner',
                       help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='W&B run name')
    
    args = parser.parse_args()
    
    # Convert string to boolean for wandb
    use_wandb = args.use_wandb.lower() == 'true'
    
    # Load configuration
    config = load_config(args.config)
    config['checkpoint_dir'] = args.checkpoint_dir
    config['device'] = args.device
    
    # Override training parameters from command line if provided
    if args.epochs is not None:
        config['epochs'] = args.epochs
        print(f"Overriding epochs from command line: {args.epochs}")
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        print(f"Overriding batch size from command line: {args.batch_size}")
    
    # Override W&B settings from command line
    config['use_wandb'] = use_wandb
    if use_wandb:
        config['wandb'] = {
            'project': args.wandb_project,
            'name': args.wandb_run_name
        }
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project=config['wandb']['project'],
            entity="francescopio",
            config=config,
            name=config['wandb'].get('name', None)
        )
    
    # Load data - handle both new and legacy formats
    if args.train_diffusion_data and args.train_true_data:
        # New format with separate train/val datasets
        train_diffusion_vis, train_true_vis = load_data(args.train_diffusion_data, args.train_true_data)
        
        if args.val_diffusion_data and args.val_true_data:
            # Separate validation dataset provided
            val_diffusion_vis, val_true_vis = load_data(args.val_diffusion_data, args.val_true_data)
            
            # Create separate datasets
            train_dataset = VisibilityDataset(train_diffusion_vis, train_true_vis)
            val_dataset = VisibilityDataset(val_diffusion_vis, val_true_vis)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers']
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers']
            )
            
            print(f"Using separate datasets:")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation samples: {len(val_dataset)}")
        else:
            # Split training data for validation
            train_loader, val_loader = create_data_loaders(train_diffusion_vis, train_true_vis, config)
            print(f"Split training data (split ratio: {config['train_split']})")
    elif args.diffusion_data and args.true_data:
        # Legacy format - single dataset with splitting
        print("Warning: Using deprecated --diffusion-data and --true-data arguments")
        diffusion_vis, true_vis = load_data(args.diffusion_data, args.true_data)
        train_loader, val_loader = create_data_loaders(diffusion_vis, true_vis, config)
    else:
        raise ValueError("Must provide either --train-diffusion-data/--train-true-data or --diffusion-data/--true-data")
    
    # Initialize trainer
    trainer = RefinerTrainer(config)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    if config.get('use_wandb', False):
        wandb.finish()

if __name__ == '__main__':
    main()