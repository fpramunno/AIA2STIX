# -*- coding: utf-8 -*-
"""
Encoder-Decoder training script for AIA data to visibility space
Maps AIA images (1, 256, 256) to visibility latent space (24, 2)

@author: francesco
"""

# import debugpy

# debugpy.connect(("v000675", 5678))  # VS Code listens on login node
# print("✅ Connected to VS Code debugger!")
# debugpy.wait_for_client()
# print("🎯 Debugger attached!")

def main():
    import argparse
    import os
    from copy import deepcopy
    import json
    from pathlib import Path
    import time
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    import numpy as np
    import wandb
    import matplotlib.pyplot as plt
    
    from src.modules import PaletteModelV2
    import src as K
    from src.data.dataset import get_aia2stix_data_objects
    
    # Parse arguments
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=32,
                   help='the batch size')
    p.add_argument('--data-path', type=str, 
                   default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
                   help='path to the AIA processed images')
    p.add_argument('--vis-path', type=str,
                   default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/vis",
                   help='path to the visibility data')
    p.add_argument('--epochs', type=int, default=100,
                   help='number of training epochs')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='learning rate')
    p.add_argument('--save-dir', type=str, default='./encoder_decoder_checkpoints',
                   help='directory to save checkpoints')
    p.add_argument('--device', type=str, default='cuda',
                   help='device to use for training')
    p.add_argument('--wandb-project', type=str, default='aia-encoder-decoder',
                   help='wandb project name')
    p.add_argument('--wandb-name', type=str, default=None,
                   help='wandb run name')
    p.add_argument('--no-wandb', action='store_true',
                   help='disable wandb logging')
    
    args = p.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity="francescopio",
            name=args.wandb_name,
            config=vars(args)
        )
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    # model = AIA_EncoderDecoder().to(device)
    model = PaletteModelV2(c_in=1, c_out=1, num_classes=None,  image_size=int(256), true_img_size=256).to(device)
    
    # Print detailed model summary
    def print_model_summary(model):
        print("\n" + "="*90)
        print("MODEL SUMMARY")
        print("="*90)
        
        total_params = 0
        trainable_params = 0
        
        print(f"{'Layer (type)':<45} {'Output Shape':<20} {'Param #':<15} {'Trainable':<10}")
        print("-"*90)
        
        # Get model structure
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # Get module type and format name
                module_type = type(module).__name__
                layer_name = f"{name} ({module_type})" if name else f"({module_type})"
                
                # Determine trainable status
                trainable_status = "Yes" if trainable > 0 else "No"
                
                if params > 0:  # Only show layers with parameters
                    print(f"{layer_name:<45} {'N/A':<20} {params:<15,} {trainable_status:<10}")
                    total_params += params
                    trainable_params += trainable
        
        print("-"*90)
        print(f"{'Total params:':<45} {'':<20} {total_params:<15,}")
        print(f"{'Trainable params:':<45} {'':<20} {trainable_params:<15,}")
        print(f"{'Non-trainable params:':<45} {'':<20} {(total_params - trainable_params):<15,}")
        print("="*90)
        
        # Architecture overview
        print("ARCHITECTURE OVERVIEW:")
        print("-"*90)
        print("Input:          AIA image (batch_size, 1, 256, 256)")
        print("Encoder:        (1, 256, 256) → (24, 2) [Visibility latent space]")
        print("                ├─ Conv2d layers with BatchNorm and ReLU")
        print("                ├─ Progressive downsampling: 256→128→64→32→16→8")
        print("                ├─ Global average pooling")
        print("                └─ Fully connected projection to latent space")
        print("")
        print("Decoder:        (24, 2) → (1, 256, 256) [Reconstructed AIA image]")
        print("                ├─ Fully connected expansion")
        print("                ├─ Reshape to feature maps")
        print("                ├─ ConvTranspose2d layers with BatchNorm and ReLU")
        print("                ├─ Progressive upsampling: 8→16→32→64→128→256")
        print("                └─ Final layer with Tanh activation")
        print("")
        print("Output:         Reconstructed AIA image (batch_size, 1, 256, 256)")
        print("                + Encoded latent representation (batch_size, 24, 2)")
        print("="*90)
        
        # Memory estimation
        print("MEMORY ESTIMATION:")
        print("-"*90)
        batch_size = 32  # Default batch size
        input_size = batch_size * 1 * 256 * 256 * 4  # float32
        latent_size = batch_size * 24 * 2 * 4  # float32
        output_size = batch_size * 1 * 256 * 256 * 4  # float32
        param_size = total_params * 4  # float32
        
        print(f"{'Parameter memory:':<30} {param_size / (1024**2):<10.2f} MB")
        print(f"{'Input memory (batch=32):':<30} {input_size / (1024**2):<10.2f} MB")
        print(f"{'Latent memory (batch=32):':<30} {latent_size / (1024**2):<10.2f} MB")
        print(f"{'Output memory (batch=32):':<30} {output_size / (1024**2):<10.2f} MB")
        print(f"{'Estimated total GPU memory:':<30} {(param_size + input_size + output_size + latent_size) / (1024**2):<10.2f} MB")
        print("="*90 + "\n")
        
        return total_params, trainable_params
    
    total_params, trainable_params = print_model_summary(model)
    
    # Log model info to wandb
    if not args.no_wandb:
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        })
        wandb.watch(model, log="all", log_freq=100)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    reconstruction_loss = nn.MSELoss()
    
    # Load data
    print("Loading data...")
    train_dataset, train_sampler, train_dl = get_aia2stix_data_objects(
        vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
        data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
        batch_size=2,
        distributed=False,
        num_data_workers=4,
        split='train',
        seed=42,
    )

    val_dataset, val_sampler, val_dl = get_aia2stix_data_objects(
        vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
        data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
        batch_size=2,
        distributed=False,
        num_data_workers=4,
        split='valid',
        seed=42,
    )
    
    train_loader = train_dl
    val_loader = val_dl
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                aia_data = batch[0].to(device)
                vis_data = batch[1].to(device)  # (batch_size, 24, 2)
            else:
                aia_data, vis_data = batch
                aia_data = aia_data.to(device)
                vis_data = vis_data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            aia_recon, _ = model(aia_data, y=None, t=torch.tensor([0], device=device))
            
            # Compute losses
            recon_loss = reconstruction_loss(aia_recon, aia_data)
            
            # Combined loss
            total_loss = recon_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(total_loss.item())
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
            })
            
            # Log to wandb every 50 batches
            if not args.no_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train/loss': total_loss.item(),
                    'train/reconstruction_loss': recon_loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for batch in pbar:
                if isinstance(batch, dict):
                    aia_data = batch[0].to(device)
                    vis_data = batch[1].to(device)
                else:
                    aia_data, vis_data = batch
                    aia_data = aia_data.to(device)
                    vis_data = vis_data.to(device)
                
                # Forward pass
                aia_recon, _ = model(aia_data, y=None, t=torch.tensor([0], device=device))
                
                # Compute losses
                recon_loss = reconstruction_loss(aia_recon, aia_data)
                total_loss = recon_loss
                
                val_losses.append(total_loss.item())
                
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                })
        
        # Calculate epoch statistics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')

        if (epoch + 1) % 5 == 0:
            aia_data_original = aia_data[0].cpu().numpy().squeeze()
            aia_recon_visual = aia_recon[0].cpu().numpy().squeeze()
            
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(aia_data_original, origin='lower')
            ax1.set_title('Original Image')
            ax2.imshow(aia_recon_visual, origin='lower')
            ax2.set_title('Predicted Image')
            plt.tight_layout()
            plt.show()
        
        # Log epoch metrics to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': avg_train_loss,
                'val/epoch_loss': avg_val_loss,
                'val/best_loss': best_val_loss,
                'Sampled images': wandb.Image(plt)
            })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  → New best model saved! (Val Loss: {best_val_loss:.4f})')
            
            # Log best model to wandb
            if not args.no_wandb:
                wandb.log({'val/new_best_loss': best_val_loss})
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Finish wandb run
    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()