#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster characteristics analysis
Shows examples from each cluster with original images and metadata
Analyzes what distinguishes the clusters based on timestamps, coordinates, and other features

@author: francesco
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from tqdm import tqdm
import warnings
from datetime import datetime
import re
import torch
from torchvision.transforms import Compose, Resize, Normalize
import math
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_encoded_data_with_filenames(data_dir):
    """Load encoded data and keep track of filenames for mapping back to original data."""
    data_path = Path(data_dir)
    
    # Look for individual sample files first
    npy_files = list(data_path.glob("*_batch_*_sample_*.npy"))
    
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No individual .npy files found in {data_dir}. Need individual files to map back to original data.")
    
    print(f"Found {len(npy_files)} individual files")
    
    # Load first file to get dimensions
    first_data = np.load(npy_files[0])
    feature_dim = first_data.shape[0] if first_data.ndim == 1 else first_data.shape[-1]
    
    # Pre-allocate array
    all_data = np.zeros((len(npy_files), feature_dim), dtype=np.float32)
    filenames = []
    
    # Load all files and keep filenames
    for i, file_path in enumerate(tqdm(npy_files, desc="Loading files")):
        data = np.load(file_path)
        if data.ndim == 1:
            all_data[i] = data
        else:
            all_data[i] = data.flatten()
        filenames.append(file_path.name)
    
    return all_data, filenames


def extract_info_from_filename(filename):
    """Extract batch and sample info from encoded filename."""
    # Expected format: train_batch_0000_sample_00.npy
    parts = filename.replace('.npy', '').split('_')
    try:
        batch_idx = int(parts[2])
        sample_idx = int(parts[4])
        return batch_idx, sample_idx
    except (IndexError, ValueError):
        return None, None


def load_original_data_mapping(processed_images_dir):
    """Create mapping from processed image files to their metadata."""
    processed_path = Path(processed_images_dir)
    aia_files = list(processed_path.glob("aia_processed_*.npy"))
    
    file_mapping = {}
    for file_path in aia_files:
        # Extract flare_id from filename: aia_processed_YYMMDDHHMMSS_rowindex.npy
        parts = file_path.stem.split('_')
        if len(parts) >= 3:
            flare_id = parts[2]  # e.g., '2105221710'
            file_mapping[file_path.name] = flare_id
    
    return file_mapping, aia_files


def load_metadata(csv_path):
    """Load the metadata CSV file."""
    df = pd.read_csv(csv_path)
    
    # Convert time_image_earth to datetime if it's not already
    if 'time_image_earth' in df.columns:
        # Handle byte strings and specific format
        df['time_image_earth'] = df['time_image_earth'].astype(str)
        # Remove 'b' prefix and quotes if present
        df['time_image_earth'] = df['time_image_earth'].str.replace("b'", "").str.replace("'", "")
        # Parse with specific format
        df['time_image_earth'] = pd.to_datetime(df['time_image_earth'], format='%d-%b-%Y %H:%M:%S.%f', errors='coerce')
    
    # Process GOES class data
    if 'GOES_class_time_of_flare' in df.columns:
        df['goes_category'] = df['GOES_class_time_of_flare'].apply(categorize_goes_class)
        df['goes_magnitude'] = df['GOES_class_time_of_flare'].apply(extract_goes_magnitude)
    
    return df


def categorize_goes_class(goes_class):
    """Categorize GOES class as 'M≤5', 'M>5', or 'X'."""
    if pd.isna(goes_class):
        return 'Unknown'
    
    goes_class = str(goes_class).upper()
    
    if goes_class.startswith('X'):
        return 'X'
    elif goes_class.startswith('M'):
        try:
            # Extract magnitude number
            magnitude = float(re.findall(r'M([0-9.]+)', goes_class)[0])
            return 'M>5' if magnitude > 5.0 else 'M≤5'
        except (IndexError, ValueError):
            return 'Unknown'
    else:
        return 'Other'


def extract_goes_magnitude(goes_class):
    """Extract numerical magnitude from GOES class."""
    if pd.isna(goes_class):
        return np.nan
    
    goes_class = str(goes_class).upper()
    
    try:
        if goes_class.startswith('X'):
            return float(re.findall(r'X([0-9.]+)', goes_class)[0])
        elif goes_class.startswith('M'):
            return float(re.findall(r'M([0-9.]+)', goes_class)[0])
        elif goes_class.startswith('C'):
            return float(re.findall(r'C([0-9.]+)', goes_class)[0])
        elif goes_class.startswith('B'):
            return float(re.findall(r'B([0-9.]+)', goes_class)[0])
        else:
            return np.nan
    except (IndexError, ValueError):
        return np.nan


def get_aia1600_transforms(target_size=256):
    """Get AIA 1600A transforms based on training/src/utils.py."""
    # AIA 1600A preprocessing config
    preprocess_config = {"min": 10, "max": 800, "scaling": "log10"}
    
    # Log10 transform with clamping
    def log_transform(x):
        return torch.log10(torch.clamp(
            torch.tensor(x, dtype=torch.float32),
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
    
    # Calculate normalization parameters
    mean = math.log10(preprocess_config["min"])
    std = math.log10(preprocess_config["max"]) - math.log10(preprocess_config["min"])
    
    # Create transform chain
    transforms = Compose([
        lambda x: log_transform(x),
        lambda x: (x - mean) / std,  # First normalization
        lambda x: (x - 0.5) / 0.5,   # Second normalization to [-1, 1]
    ])
    
    return transforms


def perform_clustering(data, n_clusters=2):
    """Perform k-means clustering on the data."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    return cluster_labels


def analyze_cluster_metadata(cluster_labels, filenames, metadata_df, original_mapping):
    """Analyze metadata characteristics for each cluster."""
    print("Analyzing cluster characteristics...")
    
    cluster_stats = {}
    
    for cluster_id in np.unique(cluster_labels):
        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster_id} ANALYSIS")
        print(f"{'='*50}")
        
        # Get samples in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_filenames = [filenames[i] for i in range(len(filenames)) if cluster_mask[i]]
        
        print(f"Number of samples: {len(cluster_filenames)}")
        
        # Try to map back to original data
        cluster_metadata = []
        successful_mappings = 0
        
        for filename in cluster_filenames:
            # Extract batch and sample indices
            batch_idx, sample_idx = extract_info_from_filename(filename)
            
            if batch_idx is not None and sample_idx is not None:
                # This is tricky - we need to map from encoded filename back to original
                # For now, let's use a simpler approach based on the order
                # This assumes the same order as in the dataset creation
                sample_global_idx = batch_idx * 16 + sample_idx  # Assuming batch size 16
                
                # Try to find corresponding metadata
                if sample_global_idx < len(metadata_df):
                    row = metadata_df.iloc[sample_global_idx]
                    cluster_metadata.append(row)
                    successful_mappings += 1
        
        print(f"Successfully mapped {successful_mappings} samples to metadata")
        
        if cluster_metadata:
            cluster_df = pd.DataFrame(cluster_metadata)
            
            # Analyze timestamps
            if 'time_image_earth' in cluster_df.columns:
                time_range = cluster_df['time_image_earth']
                print(f"Time range: {time_range.min()} to {time_range.max()}")
                
                # Group by year/month
                cluster_df['year'] = cluster_df['time_image_earth'].dt.year
                cluster_df['month'] = cluster_df['time_image_earth'].dt.month
                print(f"Year distribution: \n{cluster_df['year'].value_counts().sort_index()}")
                print(f"Month distribution: \n{cluster_df['month'].value_counts().sort_index()}")
            
            # Analyze coordinates
            if 'hpc_x_earth' in cluster_df.columns and 'hpc_y_earth' in cluster_df.columns:
                print(f"HPC X range: {cluster_df['hpc_x_earth'].min():.1f} to {cluster_df['hpc_x_earth'].max():.1f}")
                print(f"HPC Y range: {cluster_df['hpc_y_earth'].min():.1f} to {cluster_df['hpc_y_earth'].max():.1f}")
                print(f"HPC X mean: {cluster_df['hpc_x_earth'].mean():.1f} ± {cluster_df['hpc_x_earth'].std():.1f}")
                print(f"HPC Y mean: {cluster_df['hpc_y_earth'].mean():.1f} ± {cluster_df['hpc_y_earth'].std():.1f}")
            
            # Analyze roll angles
            if 'roll_angle' in cluster_df.columns:
                print(f"Roll angle range: {cluster_df['roll_angle'].min():.1f}° to {cluster_df['roll_angle'].max():.1f}°")
                print(f"Roll angle mean: {cluster_df['roll_angle'].mean():.1f}° ± {cluster_df['roll_angle'].std():.1f}°")
            
            # Analyze GOES categories  
            if 'goes_category' in cluster_df.columns:
                print(f"\nGOES Category Distribution:")
                category_counts = cluster_df['goes_category'].value_counts()
                for category, count in category_counts.items():
                    percentage = count / len(cluster_df) * 100
                    print(f"  {category}: {count} ({percentage:.1f}%)")
        
        cluster_stats[cluster_id] = {
            'n_samples': len(cluster_filenames),
            'successful_mappings': successful_mappings,
            'metadata': cluster_metadata
        }
    
    return cluster_stats


def create_cluster_examples_plot(cluster_labels, filenames, processed_images_dir, 
                                metadata_df, output_dir, n_examples=6):
    """Create a plot showing examples from each cluster with their original images."""
    print("Creating cluster examples plot...")
    
    processed_path = Path(processed_images_dir)
    n_clusters = len(np.unique(cluster_labels))
    
    # Create figure
    fig, axes = plt.subplots(n_clusters, n_examples, figsize=(4*n_examples, 4*n_clusters))
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for cluster_id in np.unique(cluster_labels):
        print(f"Processing cluster {cluster_id}...")
        
        # Get random examples from this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) < n_examples:
            selected_indices = cluster_indices
        else:
            selected_indices = np.random.choice(cluster_indices, n_examples, replace=False)
        
        for i, idx in enumerate(selected_indices):
            if i >= n_examples:
                break
                
            filename = filenames[idx]
            
            # Try to find corresponding original image
            # This is a simplified approach - in practice, you'd need more robust mapping
            batch_idx, sample_idx = extract_info_from_filename(filename)
            
            # Try to load an original image (this is approximate)
            original_files = list(processed_path.glob("aia_processed_*.npy"))
            
            if batch_idx is not None and sample_idx is not None:
                # Calculate approximate global index
                global_idx = batch_idx * 16 + sample_idx  # Assuming batch size 16
                
                if global_idx < len(original_files):
                    try:
                        # Load the image
                        img_path = original_files[global_idx]
                        img_data = np.load(img_path)
                        
                        # Apply AIA 1600A transforms for proper visualization
                        try:
                            transforms = get_aia1600_transforms()
                            img_transformed = transforms(img_data).numpy()
                            
                            # Use proper AIA 1600 colormap
                            try:
                                import matplotlib
                                sdoaia1600_cmap = matplotlib.colormaps['sdoaia1600']
                            except (KeyError, AttributeError):
                                # Fallback to hot if sdoaia1600 not available
                                sdoaia1600_cmap = 'hot'
                                print(f"Warning: sdoaia1600 colormap not available, using 'hot' as fallback")
                            
                            # Plot the image with proper normalization
                            im = axes[cluster_id, i].imshow(img_transformed, 
                                                           cmap=sdoaia1600_cmap, 
                                                           origin='lower',
                                                           vmin=-1, vmax=1)  # Normalized range
                        except Exception as transform_error:
                            print(f"Transform failed for {img_path.name}: {transform_error}")
                            # Fallback to original data with hot colormap
                            im = axes[cluster_id, i].imshow(img_data, cmap='hot', origin='lower')
                        
                        # Get metadata for title
                        if global_idx < len(metadata_df):
                            row = metadata_df.iloc[global_idx]
                            title = f"Cluster {cluster_id}\n"
                            if 'time_image_earth' in row:
                                title += f"{row['time_image_earth'].strftime('%Y-%m-%d %H:%M')}\n"
                            if 'hpc_x_earth' in row and 'hpc_y_earth' in row:
                                title += f"({row['hpc_x_earth']:.0f}, {row['hpc_y_earth']:.0f})"
                        else:
                            title = f"Cluster {cluster_id}\nSample {i+1}"
                        
                        axes[cluster_id, i].set_title(title, fontsize=8)
                        axes[cluster_id, i].set_xticks([])
                        axes[cluster_id, i].set_yticks([])
                        
                    except Exception as e:
                        print(f"Could not load image for {filename}: {e}")
                        axes[cluster_id, i].text(0.5, 0.5, f'Image\nNot Found', 
                                               ha='center', va='center', 
                                               transform=axes[cluster_id, i].transAxes)
                        axes[cluster_id, i].set_title(f"Cluster {cluster_id}\n{filename}")
            else:
                axes[cluster_id, i].text(0.5, 0.5, f'Mapping\nError', 
                                       ha='center', va='center', 
                                       transform=axes[cluster_id, i].transAxes)
                axes[cluster_id, i].set_title(f"Cluster {cluster_id}\n{filename}")
    
    # Hide unused subplots
    for cluster_id in range(n_clusters):
        for i in range(len(selected_indices), n_examples):
            axes[cluster_id, i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'cluster_examples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster examples plot saved to: {output_path}")


def create_metadata_comparison_plots(cluster_stats, metadata_df, output_dir):
    """Create plots comparing metadata characteristics between clusters."""
    print("Creating metadata comparison plots...")
    
    # Collect all metadata from all clusters
    all_cluster_metadata = {}
    for cluster_id, stats in cluster_stats.items():
        if stats['metadata']:
            all_cluster_metadata[cluster_id] = pd.DataFrame(stats['metadata'])
    
    if len(all_cluster_metadata) < 2:
        print("Not enough metadata to create comparison plots")
        return
    
    # Create comparison plots - now 3x2 grid to include GOES analysis
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    # Plot 1: Time distribution
    ax = axes[0]
    for cluster_id, cluster_df in all_cluster_metadata.items():
        if 'time_image_earth' in cluster_df.columns:
            dates = pd.to_datetime(cluster_df['time_image_earth'])
            ax.hist(dates.dt.year, alpha=0.7, label=f'Cluster {cluster_id}', bins=20)
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Temporal Distribution by Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: HPC coordinates
    ax = axes[1]
    for cluster_id, cluster_df in all_cluster_metadata.items():
        if 'hpc_x_earth' in cluster_df.columns and 'hpc_y_earth' in cluster_df.columns:
            ax.scatter(cluster_df['hpc_x_earth'], cluster_df['hpc_y_earth'], 
                      alpha=0.6, label=f'Cluster {cluster_id}', s=20)
    ax.set_xlabel('HPC X (arcsec)')
    ax.set_ylabel('HPC Y (arcsec)')
    ax.set_title('Spatial Distribution by Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Roll angle distribution
    ax = axes[2]
    for cluster_id, cluster_df in all_cluster_metadata.items():
        if 'roll_angle' in cluster_df.columns:
            ax.hist(cluster_df['roll_angle'], alpha=0.7, label=f'Cluster {cluster_id}', bins=30)
    ax.set_xlabel('Roll Angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Roll Angle Distribution by Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Monthly distribution
    ax = axes[3]
    for cluster_id, cluster_df in all_cluster_metadata.items():
        if 'time_image_earth' in cluster_df.columns:
            dates = pd.to_datetime(cluster_df['time_image_earth'])
            monthly_counts = dates.dt.month.value_counts().sort_index()
            ax.plot(monthly_counts.index, monthly_counts.values, 
                   marker='o', label=f'Cluster {cluster_id}', linewidth=2)
    ax.set_xlabel('Month')
    ax.set_ylabel('Count')
    ax.set_title('Monthly Distribution by Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))
    
    # Plot 5: GOES Category Distribution
    ax = axes[4]
    categories = ['M≤5', 'M>5', 'X', 'Other', 'Unknown']
    cluster_ids = list(all_cluster_metadata.keys())
    x_positions = np.arange(len(categories))
    width = 0.35 if len(cluster_ids) == 2 else 0.8 / len(cluster_ids)
    
    for i, cluster_id in enumerate(cluster_ids):
        cluster_df = all_cluster_metadata[cluster_id]
        if 'goes_category' in cluster_df.columns:
            category_counts = cluster_df['goes_category'].value_counts()
            counts = [category_counts.get(cat, 0) for cat in categories]
            offset = (i - len(cluster_ids)/2 + 0.5) * width
            ax.bar(x_positions + offset, counts, width, label=f'Cluster {cluster_id}', alpha=0.8)
    
    ax.set_xlabel('GOES Category')
    ax.set_ylabel('Count')
    ax.set_title('GOES Category Distribution by Cluster')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: GOES Magnitude Distribution (histogram like roll angle)
    ax = axes[5]
    for cluster_id, cluster_df in all_cluster_metadata.items():
        if 'goes_magnitude' in cluster_df.columns:
            valid_magnitudes = cluster_df['goes_magnitude'].dropna()
            if len(valid_magnitudes) > 0:
                ax.hist(valid_magnitudes, alpha=0.7, label=f'Cluster {cluster_id}', 
                       bins=20, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('GOES Magnitude')
    ax.set_ylabel('Count')
    ax.set_title('GOES Magnitude Distribution by Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'cluster_metadata_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metadata comparison plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoded-dir', type=str, required=True,
                        help='Directory containing encoded .npy files')
    parser.add_argument('--processed-images-dir', type=str, required=True,
                        help='Directory containing original processed AIA images')
    parser.add_argument('--csv-file', type=str, required=True,
                        help='Path to CSV file with metadata')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save analysis results')
    parser.add_argument('--n-examples', type=int, default=6,
                        help='Number of examples to show per cluster (default: 6)')
    parser.add_argument('--n-clusters', type=int, default=2,
                        help='Number of clusters (default: 2)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading encoded data from: {args.encoded_dir}")
    data, filenames = load_encoded_data_with_filenames(args.encoded_dir)
    print(f"Loaded {len(data)} encoded samples")
    
    print(f"Loading metadata from: {args.csv_file}")
    metadata_df = load_metadata(args.csv_file)
    print(f"Loaded metadata for {len(metadata_df)} samples")
    
    print("Performing clustering...")
    cluster_labels = perform_clustering(data, n_clusters=args.n_clusters)
    
    print("Cluster distribution:")
    for cluster_id in np.unique(cluster_labels):
        count = np.sum(cluster_labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    # Create original data mapping
    original_mapping, original_files = load_original_data_mapping(args.processed_images_dir)
    
    # Analyze cluster characteristics
    cluster_stats = analyze_cluster_metadata(cluster_labels, filenames, metadata_df, original_mapping)
    
    # Create example plots
    create_cluster_examples_plot(
        cluster_labels, filenames, args.processed_images_dir, 
        metadata_df, output_dir, n_examples=args.n_examples
    )
    
    # Create metadata comparison plots
    create_metadata_comparison_plots(cluster_stats, metadata_df, output_dir)
    
    # Save summary
    summary_file = output_dir / 'cluster_analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Cluster Characteristics Analysis Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Encoded data: {args.encoded_dir}\n")
        f.write(f"Original images: {args.processed_images_dir}\n")
        f.write(f"Metadata: {args.csv_file}\n")
        f.write(f"Number of samples: {len(data)}\n")
        f.write(f"Number of clusters: {args.n_clusters}\n\n")
        
        for cluster_id, stats in cluster_stats.items():
            f.write(f"CLUSTER {cluster_id}:\n")
            f.write(f"  Samples: {stats['n_samples']}\n")
            f.write(f"  Successful metadata mappings: {stats['successful_mappings']}\n")
            f.write(f"  Percentage: {stats['n_samples']/len(data)*100:.1f}%\n")
            
            # Add GOES class summary for each cluster
            if stats['metadata']:
                cluster_df = pd.DataFrame(stats['metadata'])
                if 'goes_category' in cluster_df.columns:
                    f.write(f"  GOES Categories:\n")
                    category_counts = cluster_df['goes_category'].value_counts()
                    for category, count in category_counts.items():
                        f.write(f"    {category}: {count} ({count/len(cluster_df)*100:.1f}%)\n")
            f.write("\n")
    
    print(f"\n{'='*60}")
    print("Cluster characteristics analysis completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - cluster_examples.png: Example images from each cluster (with AIA 1600 normalization)")
    print(f"  - cluster_metadata_comparison.png: Metadata comparisons (including GOES analysis)")
    print(f"  - cluster_analysis_summary.txt: Analysis summary")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()