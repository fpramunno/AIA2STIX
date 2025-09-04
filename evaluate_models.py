#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation Script for AIA2STIX Pipeline

This script evaluates trained models by:
1. Using either diffusion model or encoder-to-visibility model to predict visibilities
2. Using the pretrained FCD model to reconstruct images from visibilities
3. Comparing predicted vs ground truth visibilities (chi-square distance)
4. Comparing reconstructed images vs ground truth images (visual comparison)

Supports both:
- Diffusion Model → Visibilities → FCD → Images
- Encoder-to-Visibility Model → Visibilities → FCD → Images

@author: francesco
"""

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add training directory to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from src.data.dataset import get_aia2stix_data_objects
from train_encoder_to_visibility import EncoderToVisibilityModel
from util import generate_samples
import src as K

# For FCD model (Keras with configurable backend)
try:
    import os
    # Set Keras backend (can be "jax", "torch", "tensorflow")
    os.environ["KERAS_BACKEND"] = "jax"  # Default to JAX as recommended
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    print("Warning: Keras not available. FCD model functionality will be disabled.")
    KERAS_AVAILABLE = False


class FCDModelWrapper:
    """Wrapper for the FCD (Fourier Convolutional Decoder) model."""
    
    def __init__(self, model_path=None, backend="tensorflow", download_dir=None):
        if not KERAS_AVAILABLE:
            raise ImportError("Keras is required for FCD model")
        
        # Set Keras backend if specified
        if backend != os.environ.get("KERAS_BACKEND", "tensorflow"):
            os.environ["KERAS_BACKEND"] = backend
            print(f"Keras backend set to: {backend}")
        
        # Load the FCD model
        if model_path is None or model_path == "hf://mervess/FCD-Solar":
            # Download from HuggingFace hub
            try:
                print("Downloading FCD model from HuggingFace hub...")
                import huggingface_hub
                
                # Set download directory if specified
                if download_dir:
                    os.makedirs(download_dir, exist_ok=True)
                    local_model_path = huggingface_hub.snapshot_download(
                        "mervess/FCD-Solar",
                        cache_dir=download_dir
                    )
                    print(f"Model downloaded to: {local_model_path}")
                else:
                    local_model_path = huggingface_hub.snapshot_download("mervess/FCD-Solar")
                    print(f"Model downloaded to default cache: {local_model_path}")
                
                # Look for the actual .keras file in the downloaded directory
                keras_file = os.path.join(local_model_path, "fcd.keras")
                filters_file = os.path.join(local_model_path, "filters.py")
                
                if os.path.exists(keras_file) and os.path.exists(filters_file):
                    # Import GaussianFilter from the downloaded filters.py
                    import sys
                    sys.path.insert(0, local_model_path)
                    try:
                        from filters import GaussianFilter
                        custom_objects = {'GaussianFilter': GaussianFilter}
                        self.model = keras.saving.load_model(keras_file, custom_objects=custom_objects, compile=False)
                        print(f"Loaded FCD model from: {keras_file}")
                    finally:
                        sys.path.remove(local_model_path)
                else:
                    raise FileNotFoundError(f"Required files not found: {keras_file} or {filters_file}")
                print("✅ FCD model loaded successfully from HuggingFace")
                
            except ImportError:
                raise ImportError("huggingface_hub is required to download FCD model. Install with: pip install huggingface_hub")
            except Exception as e:
                print(f"Failed to download/load FCD model from HuggingFace: {e}")
                raise
        else:
            # Load from local path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"FCD model not found at: {model_path}")
            
            # Look for filters.py in the same directory
            model_dir = os.path.dirname(model_path)
            filters_file = os.path.join(model_dir, "filters.py")
            
            if os.path.exists(filters_file):
                # Import GaussianFilter from filters.py
                import sys
                sys.path.insert(0, model_dir)
                try:
                    from filters import GaussianFilter
                    custom_objects = {'GaussianFilter': GaussianFilter}
                    self.model = keras.saving.load_model(model_path, custom_objects=custom_objects, compile=False)
                finally:
                    sys.path.remove(model_dir)
            else:
                # Try loading without custom objects
                self.model = keras.saving.load_model(model_path, compile=False)
            
            print(f"FCD model loaded from: {model_path}")
        
    def predict(self, visibilities):
        """
        Predict images from visibilities.
        
        Args:
            visibilities: Array of shape (batch_size, 24, 2) - complex visibilities
            
        Returns:
            reconstructed_images: Array of shape (batch_size, 128, 128, 1)
        """
        # Convert complex visibilities to FCD input format (48 real numbers)
        if isinstance(visibilities, torch.Tensor):
            visibilities = visibilities.detach().cpu().numpy()
            
        batch_size = visibilities.shape[0]
        fcd_input = visibilities.reshape(batch_size, -1)  # (batch_size, 48)
        
        # Predict using FCD model
        reconstructed_images = self.model.predict(fcd_input, verbose=0)
        
        return reconstructed_images


def chi_square_distance(pred_vis, true_vis):
    """Calculate chi-square distance between predicted and true visibilities."""
    if isinstance(pred_vis, torch.Tensor):
        pred_vis = pred_vis.detach().cpu().numpy()
    if isinstance(true_vis, torch.Tensor):
        true_vis = true_vis.detach().cpu().numpy()
        
    pred_flat = pred_vis.reshape(-1)
    true_flat = true_vis.reshape(-1)
    
    epsilon = 1e-8
    chi_sq = np.sum((pred_flat - true_flat)**2 / (np.abs(true_flat) + epsilon))
    return chi_sq


def evaluate_diffusion_model(model, model_ema, dataloader, device, fcd_model=None):
    """Evaluate the diffusion model."""
    print("Evaluating diffusion model...")
    
    results = {
        'chi_square_distances': [],
        'predicted_visibilities': [],
        'true_visibilities': [],
        'reconstructed_images': [],
        'ground_truth_images': [],
        'original_aia_images': []
    }
    
    model_ema.eval()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating diffusion model")):
        if batch_idx >= 10:  # Limit evaluation to first 10 batches
            break
            
        # Get batch data - handle both 2 and 3 element batches
        aia_data = batch[0].contiguous().float().to(device)
        true_vis = batch[1].to(device).reshape(-1, 1, 24, 2)
        enc_vis = batch[2].to(device).reshape(-1, 1, 24, 2)
        
        
        with torch.no_grad():
            # Generate samples using the diffusion model
            samples = generate_samples(
                model_ema,
                aia_data.shape[0],
                device,
                cond_label=None,
                sampler="dpmpp_2m_sde",
                cond_img=enc_vis
            )
            
            pred_vis = samples.reshape(-1, 24, 2)
            true_vis_reshaped = true_vis.reshape(-1, 24, 2)
            
            # Calculate chi-square distance
            chi_sq = chi_square_distance(pred_vis, true_vis_reshaped)
            results['chi_square_distances'].append(chi_sq)
            
            # Store visibilities
            results['predicted_visibilities'].append(pred_vis.cpu().numpy())
            results['true_visibilities'].append(true_vis_reshaped.cpu().numpy())
            
            # Store original AIA images
            results['original_aia_images'].append(aia_data.cpu().numpy())
            
            # Generate images using FCD model if available
            if fcd_model is not None:
                # Ground truth images from true visibilities
                gt_images = fcd_model.predict(true_vis_reshaped)
                results['ground_truth_images'].append(gt_images)
                
                # Reconstructed images from predicted visibilities
                recon_images = fcd_model.predict(pred_vis)
                results['reconstructed_images'].append(recon_images)
    
    return results


def evaluate_encoder_model(model, dataloader, device, fcd_model=None):
    """Evaluate the encoder-to-visibility model."""
    print("Evaluating encoder-to-visibility model...")
    
    results = {
        'chi_square_distances': [],
        'predicted_visibilities': [],
        'true_visibilities': [],
        'reconstructed_images': [],
        'ground_truth_images': [],
        'original_aia_images': []
    }
    
    model.eval()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating encoder model")):
        if batch_idx >= 10:  # Limit evaluation to first 10 batches
            break
            
        # Get batch data
        aia_data = batch[0].float().to(device)
        true_vis = batch[1].float().to(device).reshape(-1, 24, 2)
        
        with torch.no_grad():
            # Predict visibilities directly
            pred_vis = model(aia_data)
            
            # Calculate chi-square distance
            chi_sq = chi_square_distance(pred_vis, true_vis)
            results['chi_square_distances'].append(chi_sq)
            
            # Store visibilities
            results['predicted_visibilities'].append(pred_vis.cpu().numpy())
            results['true_visibilities'].append(true_vis.cpu().numpy())
            
            # Store original AIA images
            results['original_aia_images'].append(aia_data.cpu().numpy())
            
            # Generate images using FCD model if available
            if fcd_model is not None:
                # Ground truth images from true visibilities
                gt_images = fcd_model.predict(true_vis)
                results['ground_truth_images'].append(gt_images)
                
                # Reconstructed images from predicted visibilities  
                recon_images = fcd_model.predict(pred_vis)
                results['reconstructed_images'].append(recon_images)
    
    return results


def create_comparison_plots(results, output_dir, model_name):
    """Create comprehensive comparison plots."""
    print(f"Creating comparison plots for {model_name}...")
    
    # Calculate average chi-square distance
    avg_chi_sq = np.mean(results['chi_square_distances'])
    print(f"Average chi-square distance: {avg_chi_sq:.6f}")
    
    # Create plots for first batch
    if results['predicted_visibilities'] and results['true_visibilities']:
        pred_vis = results['predicted_visibilities'][0][0]  # First sample
        true_vis = results['true_visibilities'][0][0]
        
        # Visibility comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Real and imaginary parts
        axes[0, 0].plot(pred_vis[:, 0], 'b-', label='Predicted Real', linewidth=2)
        axes[0, 0].plot(true_vis[:, 0], 'r--', label='True Real', linewidth=2)
        axes[0, 0].set_title('Real Part Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(pred_vis[:, 1], 'b-', label='Predicted Imag', linewidth=2)
        axes[0, 1].plot(true_vis[:, 1], 'r--', label='True Imag', linewidth=2)
        axes[0, 1].set_title('Imaginary Part Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Amplitude and phase
        pred_amp = np.sqrt(pred_vis[:, 0]**2 + pred_vis[:, 1]**2)
        true_amp = np.sqrt(true_vis[:, 0]**2 + true_vis[:, 1]**2)
        axes[1, 0].plot(pred_amp, 'b-', label='Predicted Amplitude', linewidth=2)
        axes[1, 0].plot(true_amp, 'r--', label='True Amplitude', linewidth=2)
        axes[1, 0].set_title('Amplitude Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        pred_phase = np.arctan2(pred_vis[:, 1], pred_vis[:, 0])
        true_phase = np.arctan2(true_vis[:, 1], true_vis[:, 0])
        axes[1, 1].plot(pred_phase, 'b-', label='Predicted Phase', linewidth=2)
        axes[1, 1].plot(true_phase, 'r--', label='True Phase', linewidth=2)
        axes[1, 1].set_title('Phase Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name} - Visibility Comparison (χ² = {avg_chi_sq:.6f})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        vis_plot_path = os.path.join(output_dir, f'{model_name.lower()}_visibility_comparison.png')
        plt.savefig(vis_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visibility comparison saved: {vis_plot_path}")
    
    # Image comparison plots if FCD results available
    if results['reconstructed_images'] and results['ground_truth_images'] and results['original_aia_images']:
        # Determine how many samples to plot (up to 10)
        num_batches = len(results['reconstructed_images'])
        max_samples_per_batch = min(10, results['reconstructed_images'][0].shape[0]) if num_batches > 0 else 0
        total_samples = min(10, num_batches * max_samples_per_batch)
        
        print(f"Creating {total_samples} image comparison plots...")
        
        sample_count = 0
        all_mse = []
        all_mae = []
        
        for batch_idx in range(num_batches):
            if sample_count >= 10:
                break
                
            recon_batch = results['reconstructed_images'][batch_idx]
            gt_batch = results['ground_truth_images'][batch_idx]
            aia_batch = results['original_aia_images'][batch_idx]
            
            batch_size = min(recon_batch.shape[0], gt_batch.shape[0], aia_batch.shape[0])
            samples_from_batch = min(batch_size, 10 - sample_count)
            
            for sample_idx in range(samples_from_batch):
                recon_img = recon_batch[sample_idx]
                gt_img = gt_batch[sample_idx]
                aia_img = aia_batch[sample_idx]

                import matplotlib
                sdoaia1600 = matplotlib.colormaps['sdoaia1600']
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original AIA image
                # Take first channel if multi-channel, normalize for display
                aia_display = aia_img[0] if len(aia_img.shape) > 2 else aia_img
                im0 = axes[0].imshow(aia_display.squeeze(), cmap=sdoaia1600, origin='lower')
                axes[0].set_title('Original AIA\n(Input Image)')
                axes[0].axis('off')
                plt.colorbar(im0, ax=axes[0], shrink=0.8)
                
                # Ground truth image
                im1 = axes[1].imshow(gt_img.squeeze(), cmap='hot', origin='lower')
                axes[1].set_title('Ground Truth\n(True Vis → FCD)')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], shrink=0.8)
                
                # Reconstructed image
                im2 = axes[2].imshow(recon_img.squeeze(), cmap='hot', origin='lower')
                axes[2].set_title(f'Reconstructed\n({model_name} → FCD)')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], shrink=0.8)
                
                # Difference
                diff = np.abs(gt_img - recon_img)
                im3 = axes[3].imshow(diff.squeeze(), cmap='viridis', origin='lower')
                axes[3].set_title('Absolute Difference')
                axes[3].axis('off')
                plt.colorbar(im3, ax=axes[3], shrink=0.8)
                
                # Calculate metrics for this sample
                mse = np.mean((gt_img - recon_img)**2)
                mae = np.mean(np.abs(gt_img - recon_img))
                all_mse.append(mse)
                all_mae.append(mae)
                
                fig.suptitle(f'{model_name} - Sample {sample_count + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}', 
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                img_plot_path = os.path.join(output_dir, f'{model_name.lower()}_image_comparison_sample_{sample_count + 1:02d}.png')
                plt.savefig(img_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                
                if sample_count >= 10:
                    break
        
        print(f"Image comparisons saved: {sample_count} plots in {output_dir}")
        
        # Print summary statistics
        if all_mse and all_mae:
            print(f"Image metrics summary over {len(all_mse)} samples:")
            print(f"  MSE - Mean: {np.mean(all_mse):.6f}, Std: {np.std(all_mse):.6f}")
            print(f"  MAE - Mean: {np.mean(all_mae):.6f}, Std: {np.std(all_mae):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Model selection
    parser.add_argument('--model-type', type=str, required=True, 
                        choices=['diffusion', 'encoder'],
                        help='Type of model to evaluate (diffusion or encoder)')
    
    # Model paths
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--config', type=str,
                        help='Path to config file (required for diffusion model)')
    parser.add_argument('--encoder-checkpoint', type=str,
                        help='Path to encoder checkpoint (required for encoder model)')
    
    # FCD model
    parser.add_argument('--fcd-model-path', type=str,
                        help='Path to the FCD model (.keras file). If not provided, downloads from HuggingFace')
    parser.add_argument('--fcd-backend', type=str, default='tensorflow',
                        choices=['jax', 'torch', 'tensorflow'],
                        help='Keras backend to use for FCD model')
    parser.add_argument('--fcd-download-dir', type=str,
                        help='Directory to download FCD model to (if downloading from HuggingFace)')
    
    # Data paths
    parser.add_argument('--data-path', type=str,
                        default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
                        help='Path to the processed AIA images')
    parser.add_argument('--vis-path', type=str,
                        default="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
                        help='Path to the visibility data CSV')
    parser.add_argument('--enc-data-path', type=str,
                        help='Path to encoded data directory (for diffusion model conditioning)')
    
    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--split', type=str, default='valid',
                        choices=['train', 'valid', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='Number of batches to evaluate (for speed)')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load FCD model if provided
    fcd_model = None
    if KERAS_AVAILABLE:
        try:
            if args.fcd_model_path:
                fcd_model = FCDModelWrapper(
                    model_path=args.fcd_model_path, 
                    backend=args.fcd_backend,
                    download_dir=args.fcd_download_dir
                )
            else:
                # Use default HuggingFace model
                fcd_model = FCDModelWrapper(
                    model_path=None,
                    backend=args.fcd_backend,
                    download_dir=args.fcd_download_dir
                )
        except Exception as e:
            print(f"Warning: Could not load FCD model: {e}")
            print("Continuing without FCD model (no image reconstruction)")
    
    # Load evaluation dataset
    print(f"Loading {args.split} dataset...")
    dataset, _, dataloader = get_aia2stix_data_objects(
        vis_path=args.vis_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=args.num_workers,
        split=args.split,
        seed=args.seed,
        enc_data_path=args.enc_data_path  # This will be None for encoder model, but used for diffusion
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Load and evaluate model based on type
    results = None
    
    if args.model_type == 'diffusion':
        # Load diffusion model
        if not args.config:
            raise ValueError("Config file is required for diffusion model")
            
        print("Loading diffusion model...")
        config = K.config.load_config(args.config)
        
        # Create model
        inner_model = K.config.make_model(config)
        inner_model_ema = inner_model  # Simplified for evaluation
        
        # Load checkpoint
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        inner_model.load_state_dict(checkpoint['model'])
        inner_model_ema.load_state_dict(checkpoint['model_ema'])
        
        # Create denoiser wrapper
        model = K.config.make_denoiser_wrapper(config)(inner_model).to(device)
        model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema).to(device)
        
        print(f"Diffusion model loaded (epoch: {checkpoint.get('epoch', 'unknown')})")
        
        # Evaluate
        results = evaluate_diffusion_model(model, model_ema, dataloader, device, fcd_model)
        model_name = "Diffusion Model"
        
    elif args.model_type == 'encoder':
        # Load encoder-to-visibility model
        if not args.encoder_checkpoint:
            raise ValueError("Encoder checkpoint is required for encoder model")
            
        print("Loading encoder-to-visibility model...")
        model = EncoderToVisibilityModel(
            encoder_checkpoint_path=args.encoder_checkpoint,
            freeze_encoder=False  # Doesn't matter for evaluation
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Encoder model loaded (epoch: {checkpoint.get('epoch', 'unknown')})")
        
        # Evaluate
        results = evaluate_encoder_model(model, dataloader, device, fcd_model)
        model_name = "Encoder Model"
    
    # Create comparison plots and save results
    if results:
        create_comparison_plots(results, str(output_dir), model_name)
        
        # Save numerical results
        results_file = output_dir / f'{args.model_type}_evaluation_results.npz'
        np.savez(
            results_file,
            chi_square_distances=results['chi_square_distances'],
            predicted_visibilities=results['predicted_visibilities'],
            true_visibilities=results['true_visibilities'],
        )
        print(f"Numerical results saved: {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Average χ² distance: {np.mean(results['chi_square_distances']):.6f}")
        print(f"Std χ² distance: {np.std(results['chi_square_distances']):.6f}")
        print(f"Results saved to: {output_dir}")
        print("="*60)


if __name__ == "__main__":
    main()