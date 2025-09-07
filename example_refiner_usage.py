#!/usr/bin/env python3
"""
Example usage of the Visibility Refiner model.

This script demonstrates how to:
1. Train a refiner model
2. Load a trained refiner
3. Use it to refine diffusion model outputs
"""

import torch
import numpy as np
from src.modules import VisibilityRefiner, ChiSquareLoss

def example_training():
    """Example of training the refiner model."""
    
    # Create synthetic data for demonstration
    n_samples = 5000
    
    # True visibilities (ground truth)
    true_vis = np.random.randn(n_samples, 24, 2) * 0.1
    
    # Diffusion model outputs (with some systematic error)
    diffusion_vis = true_vis + np.random.randn(n_samples, 24, 2) * 0.02 + 0.01
    
    # Flatten for processing
    diffusion_flat = torch.FloatTensor(diffusion_vis.reshape(-1, 48))
    true_flat = torch.FloatTensor(true_vis.reshape(-1, 48))
    
    # Initialize model
    model = VisibilityRefiner(
        input_dim=48,
        hidden_dims=[64, 32],  # Small model
        output_dim=48,
        dropout_rate=0.1,
        use_residual=True
    )
    
    # Loss function and optimizer
    criterion = ChiSquareLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        refined_vis = model(diffusion_flat)
        loss = criterion(refined_vis, true_flat)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model

def example_inference(model):
    """Example of using trained refiner for inference."""
    
    # Sample diffusion model output
    diffusion_output = torch.randn(10, 48)  # 10 samples, 48 dimensions each
    
    model.eval()
    with torch.no_grad():
        refined_output = model(diffusion_output)
    
    print(f"Input shape: {diffusion_output.shape}")
    print(f"Output shape: {refined_output.shape}")
    print(f"Refinement (first sample): {(refined_output[0] - diffusion_output[0])[:5]}")

def main():
    print("Training refiner model...")
    trained_model = example_training()
    
    print("\nUsing trained model for inference...")
    example_inference(trained_model)
    
    print("\nExample complete!")
    print("\nTo train on real data, use:")
    print("python train_visibility_refiner.py --config configs/refiner_config.yaml \\")
    print("    --diffusion-data your_diffusion_outputs.npy \\")
    print("    --true-data your_true_visibilities.npy")
    
    print("\nTo test a trained model:")
    print("python test_refiner.py --checkpoint checkpoints/best_model.pth \\")
    print("    --diffusion-data your_test_diffusion_outputs.npy \\")
    print("    --true-data your_test_true_visibilities.npy")

if __name__ == '__main__':
    main()