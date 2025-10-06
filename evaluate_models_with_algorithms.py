#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Model Evaluation Script for AIA2STIX Pipeline

This script evaluates trained models with selectable reconstruction algorithms:
1. Using either diffusion model or encoder-to-visibility model to predict visibilities
2. Using selected reconstruction algorithms: FCD, Clean, EM, MEM GE, Back Projection
3. Comparing predicted vs ground truth visibilities (chi-square distance)
4. Comparing reconstructed images vs ground truth images (visual comparison)

Supports both:
- Diffusion Model → Visibilities → [Selected Algorithms] → Images
- Encoder-to-Visibility Model → Visibilities → [Selected Algorithms] → Images

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
from src.utils import get_alpha

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

# For STIX algorithms
try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
    from sunpy.map import Map, make_fitswcs_header
    from xrayvision.clean import vis_clean
    from xrayvision.imaging import vis_to_image, vis_to_map
    from xrayvision.mem import mem, resistant_mean
    STIX_AVAILABLE = True
except ImportError:
    print("Warning: STIX dependencies not available. STIX algorithms will be disabled.")
    print("Install with: pip install sunpy stixpy xrayvision")
    STIX_AVAILABLE = False


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
        fcd_input = 2*visibilities.reshape(batch_size, -1)  # (batch_size, 48)
        
        # Predict using FCD model
        reconstructed_images = self.model.predict(fcd_input, verbose=0)
        
        return reconstructed_images


class STIXAlgorithmsWrapper:
    """Wrapper for STIX imaging algorithms."""
    
    def __init__(self, imsize=[128, 128], pixel_size=2.5):
        self.imsize = imsize * u.pixel if STIX_AVAILABLE else imsize
        self.pixel_size = [pixel_size, pixel_size] * u.arcsec / u.pixel if STIX_AVAILABLE else pixel_size
        self.algorithms = ['back_projection', 'clean', 'mem', 'em']
        
        if STIX_AVAILABLE:
            # Get real STIX u,v coordinates
            try:
                from stixpy.calibration.visibility import get_uv_points_data
                self.uv_data = get_uv_points_data()
                self.use_real_stix = True
                print("Using real STIX u,v coordinates and proper visibility objects")
            except Exception as e:
                print(f"Could not load STIX u,v data: {e}")
                self.use_real_stix = False
                self.use_simplified = True
        else:
            self.use_real_stix = False
            self.use_simplified = True
            print("STIX not available, using simplified algorithms")
        
    def create_stix_visibility_object(self, visibilities):
        """Create proper STIX visibility object using real u,v coordinates."""
        if self.use_real_stix:
            # Convert [real, imag] to complex
            if visibilities.shape[-1] == 2:
                complex_vis = visibilities[:, 0] + 1j * visibilities[:, 1]
            else:
                complex_vis = visibilities
                
            # Use real STIX u,v coordinates (24 subcollimators)
            u_coords = self.uv_data['u'][:24]  # arcsec^-1
            v_coords = self.uv_data['v'][:24]  # arcsec^-1
            isc_indices = self.uv_data['isc'][:24]
            
            # Create proper uncertainty estimates
            amplitude_uncertainty = 0.05 * np.abs(complex_vis) + 1e-6  # 5% + floor
            
            # Create proper Visibilities object
            from xrayvision.visibility import Visibilities, VisMeta
            from astropy.time import Time
            
            vis_meta = VisMeta(
                instrumet="STIX",
                spectral_range=[25, 28] * u.keV,
                time_range=Time(['2021-05-22T17:09:25', '2021-05-22T17:10:25']),
                vis_labels=[f"det_{i}" for i in isc_indices],
                isc=isc_indices.value,
                calibrated=True,
            )
            
            # Create phase center
            phase_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=Helioprojective)
            vis_meta['offset'] = phase_center
            
            vis_obj = Visibilities(
                complex_vis[:24] * u.dimensionless_unscaled,  # Visibilities also need units
                u=u_coords,
                v=v_coords,
                amplitude=np.abs(complex_vis[:24]) * u.dimensionless_unscaled,
                amplitude_uncertainty=amplitude_uncertainty[:24] * u.dimensionless_unscaled,
                meta=vis_meta,
            )
            
            return vis_obj
            
        else:
            # Fallback to simplified mock object
            return self._create_mock_visibility_object(visibilities)
            
    def _create_mock_visibility_object(self, visibilities):
        """Create simplified mock visibility object."""
        class MockVisibility:
            def __init__(self, vis_data):
                # Convert [real, imag] to complex
                if vis_data.shape[-1] == 2:
                    self.visibilities = vis_data[:, 0] + 1j * vis_data[:, 1]
                else:
                    self.visibilities = vis_data
                    
                # Mock uncertainty
                self.amplitude_uncertainty = 0.1 * np.abs(self.visibilities) + 1e-6
                
                # Simple mock u,v coordinates
                n_vis = len(self.visibilities)
                angles = np.linspace(0, 2*np.pi, n_vis, endpoint=False)
                radii = np.linspace(5, 50, n_vis)  # Different spatial frequencies
                
                self.u = radii * np.cos(angles)
                self.v = radii * np.sin(angles)
                
        return MockVisibility(visibilities)
    
    def predict_single_algorithm(self, visibilities, algorithm, alpha_values=None):
        """Predict using a single algorithm."""
        if isinstance(visibilities, torch.Tensor):
            visibilities = visibilities.detach().cpu().numpy()

        if isinstance(alpha_values, torch.Tensor):
            alpha_values = alpha_values.detach().cpu().numpy()

        batch_size = visibilities.shape[0]
        results = []

        for i in range(batch_size):
            vis_sample = visibilities[i]  # Shape: (24, 2)

            # IMPORTANT: Denormalize visibilities for STIX algorithms
            # The dataset normalizes by: vis_normalized = (vis / alpha) / 2
            # We need to reverse this: vis_original = vis_normalized * 2 * alpha
            if alpha_values is not None and alpha_values[i] > 0:
                print(f'Alpha is: {alpha_values[i]}', flush=True)
                alpha = alpha_values[i]
                vis_denormalized = vis_sample * 2 * alpha
            else:
                # Fallback if alpha not provided (shouldn't happen)
                vis_denormalized = vis_sample
            
            if self.use_real_stix:
                # Use real STIX algorithms with proper visibility objects
                result = self._real_stix_reconstruction(vis_denormalized, algorithm)
            else:
                # Simplified version using basic transforms
                result = self._simplified_reconstruction(vis_denormalized, algorithm)
                
            results.append(result)
            
        return np.array(results)
    
    def _real_stix_reconstruction(self, visibilities, algorithm):
        """Real STIX reconstruction using proper visibility objects."""
        try:
            # Create proper STIX visibility object
            vis_obj = self.create_stix_visibility_object(visibilities)
            
            if algorithm == 'back_projection':
                image = vis_to_image(vis_obj, self.imsize, pixel_size=self.pixel_size)
            elif algorithm == 'clean':
                clean_map, _, _ = vis_clean(
                    vis_obj, self.imsize, pixel_size=self.pixel_size,
                    gain=0.1, niter=100, clean_beam_width=20 * u.arcsec
                )
                image = clean_map.data
            elif algorithm == 'mem':
                snr_value, _ = resistant_mean(
                    (np.abs(vis_obj.visibilities) / vis_obj.amplitude_uncertainty).flatten(), 3
                )
                percent_lambda = (2 / (snr_value**2 + 90)) * u.dimensionless_unscaled
                mem_map = mem(vis_obj, shape=self.imsize, 
                            pixel_size=self.pixel_size, percent_lambda=percent_lambda)
                image = mem_map.data
            elif algorithm == 'em':
                # For EM, we need to create mock meta pixels
                # This is a simplified approach since we don't have real ABCD data
                mock_abcd = np.random.poisson(100, (24, 4)) * 1e-3  # Mock ABCD rates
                from stixpy.imaging.em import em
                em_map = em(
                    mock_abcd,
                    vis_obj,
                    shape=self.imsize,
                    pixel_size=self.pixel_size,
                    flare_location=vis_obj.meta['offset'],
                    idx=np.arange(24),
                )
                image = em_map
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Ensure proper size
            if hasattr(image, 'shape') and image.shape != (128, 128):
                from scipy.ndimage import zoom
                image = zoom(image, (128/image.shape[0], 128/image.shape[1]))
                
            return image[..., np.newaxis] if image.ndim == 2 else image
            
        except Exception as e:
            print(f"Real STIX algorithm {algorithm} failed: {e}")
            # Fallback to simplified
            return self._simplified_reconstruction(visibilities, algorithm)
    
    def _simplified_reconstruction(self, visibilities, algorithm):
        """Simplified reconstruction for when STIX is not available."""
        # Convert to complex
        if visibilities.shape[-1] == 2:
            complex_vis = visibilities[:, 0] + 1j * visibilities[:, 1]
        else:
            complex_vis = visibilities
            
        # Create a proper 2D inverse FFT reconstruction
        # Use a more sophisticated approach than simple grid placement
        
        if algorithm == 'back_projection':
            image = self._back_projection_simple(complex_vis)
        elif algorithm == 'clean':
            image = self._clean_simple(complex_vis)
        elif algorithm == 'mem':
            image = self._mem_simple(complex_vis)
        elif algorithm == 'em':
            image = self._em_simple(complex_vis)
        else:
            image = self._back_projection_simple(complex_vis)
        
        # Ensure proper normalization
        if image.max() > 0:
            image = image / image.max()
            
        return image[..., np.newaxis]
    
    def _back_projection_simple(self, complex_vis):
        """Simple back projection using inverse FFT."""
        # Create 2D grid from 1D visibilities
        grid_size = 128
        vis_grid = np.zeros((grid_size, grid_size), dtype=complex)
        
        # Place visibilities in a radial pattern (more realistic than grid)
        center = grid_size // 2
        n_vis = len(complex_vis)
        
        for i, vis in enumerate(complex_vis):
            # Radial placement with some randomness
            angle = 2 * np.pi * i / n_vis
            radius = min(10 + i, center - 5)  # Vary radius
            
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            
            if 0 <= x < grid_size and 0 <= y < grid_size:
                vis_grid[y, x] = vis
        
        # Inverse FFT
        image = np.abs(np.fft.ifft2(np.fft.ifftshift(vis_grid)))
        return image
    
    def _clean_simple(self, complex_vis):
        """Simple approximation of CLEAN algorithm."""
        # Start with back projection
        image = self._back_projection_simple(complex_vis)
        
        # Apply iterative deconvolution-like process
        for _ in range(10):  # Few iterations
            # Find peak
            peak_idx = np.unravel_index(np.argmax(image), image.shape)
            peak_val = image[peak_idx]
            
            if peak_val < 0.01 * image.max():
                break
                
            # Subtract a fraction of the peak with Gaussian
            from scipy.ndimage import gaussian_filter
            gaussian_peak = np.zeros_like(image)
            gaussian_peak[peak_idx] = peak_val * 0.1  # Clean gain
            gaussian_component = gaussian_filter(gaussian_peak, sigma=2.0)
            
            image -= gaussian_component
            image = np.maximum(image, 0)  # Keep positive
        
        return image
    
    def _mem_simple(self, complex_vis):
        """Simple approximation of MEM algorithm."""
        # Start with back projection
        image = self._back_projection_simple(complex_vis)
        
        # Apply entropy-like regularization (simplified)
        from scipy.ndimage import gaussian_filter
        
        # Smooth the image (entropy tends to prefer smooth solutions)
        image = gaussian_filter(image, sigma=1.5)
        
        # Apply non-linear transformation similar to MEM
        image = np.maximum(image, 1e-6)  # Ensure positive
        image = image ** 0.8  # Non-linear transformation
        
        return image
    
    def _em_simple(self, complex_vis):
        """Simple approximation of EM algorithm."""
        # Start with uniform image
        image = np.ones((128, 128)) * np.abs(complex_vis).mean()
        
        # Simple iterative improvement (mock EM steps)
        bp_image = self._back_projection_simple(complex_vis)
        
        # Weighted combination emphasizing point sources
        for _ in range(5):
            # Update step (simplified EM update)
            correction = bp_image / (image + 1e-6)
            image *= correction ** 0.1  # Small step size
            
            # Apply Poisson-like constraint
            image = np.maximum(image, 1e-6)
            
            # Slight smoothing
            from scipy.ndimage import gaussian_filter
            image = gaussian_filter(image, sigma=0.5)
        
        return image
    
    def _full_stix_reconstruction(self, visibilities, algorithm):
        """Full STIX reconstruction."""
        try:
            vis_obj = self.create_mock_visibility_object(visibilities)
            
            if algorithm == 'back_projection':
                image = vis_to_image(vis_obj, self.imsize, pixel_size=self.pixel_size)
            elif algorithm == 'clean':
                clean_map, _, _ = vis_clean(
                    vis_obj, self.imsize, pixel_size=self.pixel_size,
                    gain=0.1, niter=100, clean_beam_width=20 * u.arcsec
                )
                image = clean_map.data
            elif algorithm == 'mem':
                snr_value, _ = resistant_mean(
                    (np.abs(vis_obj.visibilities) / vis_obj.amplitude_uncertainty).flatten(), 3
                )
                # Add units to percent_lambda as required by astropy
                percent_lambda = (2 / (snr_value**2 + 90)) * u.dimensionless_unscaled
                mem_map = mem(vis_obj, shape=self.imsize, 
                            pixel_size=self.pixel_size, percent_lambda=percent_lambda)
                image = mem_map.data
            elif algorithm == 'em':
                # Simplified EM (use back projection as placeholder)
                image = vis_to_image(vis_obj, self.imsize, pixel_size=self.pixel_size)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Resize to 128x128 if needed
            if image.shape != (128, 128):
                from scipy.ndimage import zoom
                image = zoom(image, (128/image.shape[0], 128/image.shape[1]))
                
            return image[..., np.newaxis]
            
        except Exception as e:
            print(f"STIX algorithm {algorithm} failed: {e}")
            return np.zeros((128, 128, 1))


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


def evaluate_with_algorithms(model, model_ema, dataloader, device, 
                           reconstruction_models, model_type='diffusion'):
    """Evaluate model with multiple reconstruction algorithms."""
    print(f"Evaluating {model_type} model with {len(reconstruction_models)} reconstruction algorithms...")
    
    results = {
        'chi_square_distances': [],
        'predicted_visibilities': [],
        'true_visibilities': [],
        'original_aia_images': [],
        'reconstructed_images': {alg: [] for alg in reconstruction_models.keys()},
        'ground_truth_images': {alg: [] for alg in reconstruction_models.keys()}
    }
    
    if model_type == 'diffusion':
        model_ema.eval()
    else:
        model.eval()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_type} model")):
        if batch_idx >= 10:  # Limit evaluation to first 10 batches
            break
            
        # Get batch data
        aia_data = batch[0].contiguous().float().to(device)
        true_vis = batch[1].to(device).reshape(-1, 24, 2)

        if model_type == 'diffusion':
            enc_vis = batch[2].to(device).reshape(-1, 1, 24, 2)
            alpha_values = batch[3].to(device)  # Alpha values for denormalization
        else:
            alpha_values = batch[2].to(device)  # For encoder, alpha is at index 2
        
        with torch.no_grad():
            # Predict visibilities
            if model_type == 'diffusion':
                samples = generate_samples(
                    model_ema,
                    aia_data.shape[0],
                    device,
                    cond_label=None,
                    sampler="dpmpp_2m_sde",
                    cond_img=enc_vis
                )
                pred_vis = samples.reshape(-1, 24, 2)
            else:  # encoder
                pred_vis = model(aia_data)
            
            # Calculate chi-square distance
            chi_sq = chi_square_distance(pred_vis, true_vis)
            results['chi_square_distances'].append(chi_sq)
            
            # Store visibilities and AIA images
            results['predicted_visibilities'].append(pred_vis.cpu().numpy())
            results['true_visibilities'].append(true_vis.cpu().numpy())
            results['original_aia_images'].append(aia_data.cpu().numpy())
            
            # Generate images using each reconstruction algorithm
            for alg_name, model_wrapper in reconstruction_models.items():
                try:
                    # Ground truth images
                    if hasattr(model_wrapper, 'predict_single_algorithm'):
                        # STIX algorithm - pass alpha values for denormalization
                        gt_images = model_wrapper.predict_single_algorithm(true_vis, alg_name.split('_')[-1], alpha_values)
                        recon_images = model_wrapper.predict_single_algorithm(pred_vis, alg_name.split('_')[-1], alpha_values)
                    else:
                        # FCD model
                        gt_images = model_wrapper.predict(true_vis)
                        recon_images = model_wrapper.predict(pred_vis)
                    
                    results['ground_truth_images'][alg_name].append(gt_images)
                    results['reconstructed_images'][alg_name].append(recon_images)
                    
                except Exception as e:
                    print(f"Algorithm {alg_name} failed: {e}")
                    # Add zeros for failed reconstructions
                    batch_size = pred_vis.shape[0]
                    zero_images = np.zeros((batch_size, 128, 128, 1))
                    results['ground_truth_images'][alg_name].append(zero_images)
                    results['reconstructed_images'][alg_name].append(zero_images)
    
    return results


def create_algorithm_comparison_plots(results, output_dir, model_name, algorithms):
    """Create comprehensive comparison plots for all algorithms."""
    print(f"Creating comparison plots for {model_name} with {len(algorithms)} algorithms...")
    
    # Calculate average chi-square distance
    avg_chi_sq = np.mean(results['chi_square_distances'])
    print(f"Average chi-square distance: {avg_chi_sq:.6f}")
    
    # Visibility comparison plot (unchanged)
    if results['predicted_visibilities'] and results['true_visibilities']:
        pred_vis = results['predicted_visibilities'][0][0]
        true_vis = results['true_visibilities'][0][0]
        
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
        
    # Multi-algorithm image comparison plots
    if all(results['reconstructed_images'][alg] for alg in algorithms):
        num_algorithms = len(algorithms)
        max_samples = 5  # Show fewer samples due to more algorithms
        
        for sample_idx in range(min(max_samples, len(results['original_aia_images']))):
            # Get sample data
            aia_img = results['original_aia_images'][sample_idx][0]  # First sample in batch
            
            # Create subplot grid: AIA + algorithms (2 rows)
            cols = num_algorithms + 1  # +1 for AIA input
            rows = 2  # Ground Truth + Predicted
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            if cols == 1:
                axes = axes.reshape(-1, 1)
            
            # AIA input (both rows, first column)
            import matplotlib
            sdoaia1600 = matplotlib.colormaps['sdoaia1600']
            aia_display = aia_img[0] if len(aia_img.shape) > 2 else aia_img
            
            # Show AIA in both rows
            axes[0, 0].imshow(aia_display.squeeze(), cmap=sdoaia1600, origin='lower')
            axes[0, 0].set_title('AIA Input')
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(aia_display.squeeze(), cmap=sdoaia1600, origin='lower')
            axes[1, 0].set_title('AIA Input')
            axes[1, 0].axis('off')
            
            # Ground truth images (row 0, cols 1+)
            for col, alg in enumerate(algorithms, 1):
                gt_img = results['ground_truth_images'][alg][sample_idx//len(results['ground_truth_images'][alg])][sample_idx%10]
                axes[0, col].imshow(gt_img.squeeze(), cmap='hot', origin='lower')
                axes[0, col].set_title(f'GT - {alg.upper().replace("STIX_", "")}')
                axes[0, col].axis('off')
                
            # Predicted images (row 1, cols 1+)
            all_mse = {}
            for col, alg in enumerate(algorithms, 1):
                recon_img = results['reconstructed_images'][alg][sample_idx//len(results['reconstructed_images'][alg])][sample_idx%10]
                gt_img = results['ground_truth_images'][alg][sample_idx//len(results['ground_truth_images'][alg])][sample_idx%10]
                
                axes[1, col].imshow(recon_img.squeeze(), cmap='hot', origin='lower')
                
                # Calculate MSE for this algorithm
                mse = np.mean((gt_img - recon_img)**2)
                all_mse[alg] = mse
                
                axes[1, col].set_title(f'PRED - {alg.upper().replace("STIX_", "")}\nMSE: {mse:.4f}')
                axes[1, col].axis('off')
            
            # No vertical row labels - cleaner look
            
            fig.suptitle(f'{model_name} - Algorithm Comparison - Sample {sample_idx + 1}', 
                         fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            img_plot_path = os.path.join(output_dir, f'{model_name.lower()}_algorithms_comparison_sample_{sample_idx + 1:02d}.png')
            plt.savefig(img_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Print MSE comparison
            print(f"Sample {sample_idx + 1} MSE comparison:")
            for alg, mse in sorted(all_mse.items(), key=lambda x: x[1]):
                print(f"  {alg}: {mse:.6f}")


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
    
    # Reconstruction algorithms selection
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        choices=['fcd', 'clean', 'mem', 'em', 'back_projection'],
                        default=['fcd'],
                        help='Reconstruction algorithms to use')
    
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
    print(f"Selected algorithms: {args.algorithms}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load reconstruction models based on selected algorithms
    reconstruction_models = {}
    
    # Load FCD model if selected
    if 'fcd' in args.algorithms and KERAS_AVAILABLE:
        try:
            reconstruction_models['fcd'] = FCDModelWrapper(
                model_path=args.fcd_model_path,
                backend=args.fcd_backend,
                download_dir=args.fcd_download_dir
            )
            print("✅ FCD model loaded")
        except Exception as e:
            print(f"❌ Could not load FCD model: {e}")
    
    # Load STIX algorithms if selected
    stix_algorithms = [alg for alg in args.algorithms if alg in ['clean', 'mem', 'em', 'back_projection']]
    if stix_algorithms:
        stix_wrapper = STIXAlgorithmsWrapper()
        for alg in stix_algorithms:
            reconstruction_models[f'stix_{alg}'] = stix_wrapper
        print(f"✅ STIX algorithms loaded: {stix_algorithms}")
    
    if not reconstruction_models:
        raise ValueError("No reconstruction models could be loaded!")
    
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
        enc_data_path=args.enc_data_path
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Load and evaluate model
    if args.model_type == 'diffusion':
        if not args.config:
            raise ValueError("Config file is required for diffusion model")
            
        print("Loading diffusion model...")
        config = K.config.load_config(args.config)
        
        inner_model = K.config.make_model(config)
        inner_model_ema = inner_model
        
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        inner_model.load_state_dict(checkpoint['model'])
        inner_model_ema.load_state_dict(checkpoint['model_ema'])
        
        model = K.config.make_denoiser_wrapper(config)(inner_model).to(device)
        model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema).to(device)
        
        print(f"Diffusion model loaded (epoch: {checkpoint.get('epoch', 'unknown')})")
        
        results = evaluate_with_algorithms(model, model_ema, dataloader, device, 
                                         reconstruction_models, 'diffusion')
        model_name = "Diffusion Model"
        
    elif args.model_type == 'encoder':
        if not args.encoder_checkpoint:
            raise ValueError("Encoder checkpoint is required for encoder model")
            
        print("Loading encoder-to-visibility model...")
        model = EncoderToVisibilityModel(
            encoder_checkpoint_path=args.encoder_checkpoint,
            freeze_encoder=False
        ).to(device)
        
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Encoder model loaded (epoch: {checkpoint.get('epoch', 'unknown')})")
        
        results = evaluate_with_algorithms(model, None, dataloader, device, 
                                         reconstruction_models, 'encoder')
        model_name = "Encoder Model"
    
    # Create comparison plots
    if results:
        create_algorithm_comparison_plots(results, str(output_dir), model_name, list(reconstruction_models.keys()))

        # Save numerical results
        results_file = output_dir / f'{args.model_type}_algorithms_evaluation_results.npz'

        # Concatenate batches into single arrays
        pred_vis_all = np.concatenate(results['predicted_visibilities'], axis=0)
        true_vis_all = np.concatenate(results['true_visibilities'], axis=0)
        chi_sq_all = np.array(results['chi_square_distances'])
        aia_images_all = np.concatenate(results['original_aia_images'], axis=0)

        # Concatenate reconstructed images for each algorithm
        recon_images_dict = {}
        gt_images_dict = {}
        for alg_name in reconstruction_models.keys():
            recon_images_dict[f'reconstructed_{alg_name}'] = np.concatenate(results['reconstructed_images'][alg_name], axis=0)
            gt_images_dict[f'ground_truth_{alg_name}'] = np.concatenate(results['ground_truth_images'][alg_name], axis=0)

        print(f"\nSaving numerical results...")
        print(f"  Predicted visibilities shape: {pred_vis_all.shape}")
        print(f"  True visibilities shape: {true_vis_all.shape}")
        print(f"  Chi-square distances shape: {chi_sq_all.shape}")
        print(f"  AIA images shape: {aia_images_all.shape}")
        for alg_name in reconstruction_models.keys():
            print(f"  {alg_name} reconstructed images shape: {recon_images_dict[f'reconstructed_{alg_name}'].shape}")

        np.savez(
            results_file,
            chi_square_distances=chi_sq_all,
            predicted_visibilities=pred_vis_all,
            true_visibilities=true_vis_all,
            original_aia_images=aia_images_all,
            **recon_images_dict,
            **gt_images_dict
        )
        print(f"Numerical results saved: {results_file}")

        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Algorithms: {list(reconstruction_models.keys())}")
        print(f"Average χ² distance: {np.mean(results['chi_square_distances']):.6f}")
        print(f"Results saved to: {output_dir}")
        print("="*80)


if __name__ == "__main__":
    main()