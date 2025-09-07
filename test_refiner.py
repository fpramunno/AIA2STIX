#!/usr/bin/env python3

import torch
import numpy as np
import argparse
from pathlib import Path
import yaml
from src.modules import VisibilityRefiner, ChiSquareLoss

def load_model(checkpoint_path, config_path=None):
    """Load trained refiner model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['model']
    else:
        config = checkpoint['config']['model']
    
    model = VisibilityRefiner(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        dropout_rate=config['dropout_rate'],
        use_residual=config['use_residual']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def evaluate_refiner(model, diffusion_vis, true_vis, device='cpu'):
    """Evaluate refiner performance."""
    model = model.to(device)
    
    # Flatten visibilities
    diffusion_flat = diffusion_vis.reshape(-1, 48)
    true_flat = true_vis.reshape(-1, 48)
    
    # Convert to tensors
    diffusion_tensor = torch.FloatTensor(diffusion_flat).to(device)
    true_tensor = torch.FloatTensor(true_flat).to(device)
    
    with torch.no_grad():
        refined_tensor = model(diffusion_tensor)
    
    # Calculate metrics
    mse_loss = torch.nn.MSELoss()
    chi_square_loss = ChiSquareLoss()
    
    # Before refinement
    diffusion_mse = mse_loss(diffusion_tensor, true_tensor).item()
    diffusion_chi2 = chi_square_loss(diffusion_tensor, true_tensor).item()
    
    # After refinement
    refined_mse = mse_loss(refined_tensor, true_tensor).item()
    refined_chi2 = chi_square_loss(refined_tensor, true_tensor).item()
    
    print(f"Before Refinement:")
    print(f"  MSE Loss: {diffusion_mse:.6f}")
    print(f"  Chi-Square Loss: {diffusion_chi2:.6f}")
    
    print(f"\nAfter Refinement:")
    print(f"  MSE Loss: {refined_mse:.6f}")
    print(f"  Chi-Square Loss: {refined_chi2:.6f}")
    
    print(f"\nImprovement:")
    print(f"  MSE: {((diffusion_mse - refined_mse) / diffusion_mse * 100):.2f}%")
    print(f"  Chi-Square: {((diffusion_chi2 - refined_chi2) / diffusion_chi2 * 100):.2f}%")
    
    return {
        'diffusion_mse': diffusion_mse,
        'diffusion_chi2': diffusion_chi2,
        'refined_mse': refined_mse,
        'refined_chi2': refined_chi2,
        'refined_visibilities': refined_tensor.cpu().numpy().reshape(-1, 24, 2)
    }

def create_synthetic_data(n_samples=1000):
    """Create synthetic visibility data for testing."""
    # Create some synthetic true visibilities
    true_vis = np.random.randn(n_samples, 24, 2) * 0.1
    
    # Create diffusion outputs by adding noise to true visibilities
    noise_level = 0.05
    diffusion_vis = true_vis + np.random.randn(n_samples, 24, 2) * noise_level
    
    return diffusion_vis, true_vis

def main():
    parser = argparse.ArgumentParser(description='Test Visibility Refiner')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to config file (optional, will use config from checkpoint)')
    parser.add_argument('--diffusion-data', type=str,
                       help='Path to diffusion model visibility outputs (.npy)')
    parser.add_argument('--true-data', type=str,
                       help='Path to true visibilities (.npy)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--output', type=str,
                       help='Path to save refined visibilities (.npy)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.config)
    print(f"Model loaded successfully from {args.checkpoint}")
    
    # Load or create data
    if args.synthetic:
        print("Creating synthetic data...")
        diffusion_vis, true_vis = create_synthetic_data(1000)
    else:
        if not args.diffusion_data or not args.true_data:
            raise ValueError("Must provide --diffusion-data and --true-data or use --synthetic")
        
        print("Loading real data...")
        diffusion_vis = np.load(args.diffusion_data)
        true_vis = np.load(args.true_data)
    
    print(f"Data shape: {diffusion_vis.shape}")
    
    # Evaluate model
    print("\nEvaluating refiner...")
    results = evaluate_refiner(model, diffusion_vis, true_vis, args.device)
    
    # Save refined visibilities if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, results['refined_visibilities'])
        print(f"\nRefined visibilities saved to {output_path}")

if __name__ == '__main__':
    main()