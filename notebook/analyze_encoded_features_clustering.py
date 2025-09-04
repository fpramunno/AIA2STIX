#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering-enhanced analysis of encoded features
Includes k-means clustering with elbow method and color-coded visualizations
Analyzes relationship between clustering and dimensionality reduction parameters

@author: francesco
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import dimensionality reduction and clustering libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")


def load_encoded_data(data_dir):
    """Load all .npy files from directory and concatenate."""
    data_path = Path(data_dir)
    
    # Look for individual sample files first
    npy_files = list(data_path.glob("*_batch_*_sample_*.npy"))
    
    if len(npy_files) == 0:
        # Try to load concatenated file if individual files not found
        concat_files = list(data_path.glob("*_encoded_features.npy"))
        if len(concat_files) > 0:
            print(f"Loading concatenated file: {concat_files[0]}")
            return np.load(concat_files[0])
        else:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")
    
    print(f"Found {len(npy_files)} individual files")
    
    # Load first file to get dimensions
    first_data = np.load(npy_files[0])
    feature_dim = first_data.shape[0] if first_data.ndim == 1 else first_data.shape[-1]
    
    # Pre-allocate array
    all_data = np.zeros((len(npy_files), feature_dim), dtype=np.float32)
    
    # Load all files
    for i, file_path in enumerate(tqdm(npy_files, desc="Loading files")):
        data = np.load(file_path)
        if data.ndim == 1:
            all_data[i] = data
        else:
            all_data[i] = data.flatten()
    
    return all_data


def find_optimal_clusters(data, output_dir, max_clusters=15):
    """Find optimal number of clusters using elbow method and silhouette analysis."""
    print("Finding optimal number of clusters...")
    
    # Standardize the data for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Test different numbers of clusters
    k_range = range(2, min(max_clusters + 1, len(data) // 2))  # Ensure we don't exceed data size
    inertias = []
    silhouette_scores = []
    
    print(f"Testing k from 2 to {max(k_range)}")
    
    for k in tqdm(k_range, desc="Computing clustering metrics"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(data_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
    
    # Find elbow using simple method (biggest drop in inertia)
    if len(inertias) > 2:
        # Calculate rate of change
        deltas = np.diff(inertias)
        delta_deltas = np.diff(deltas)
        # Find point where second derivative is maximum (biggest change in slope)
        elbow_idx = np.argmax(delta_deltas) + 2  # +2 because of double diff and 0-based indexing
        optimal_k_elbow = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[len(k_range)//2]
    else:
        optimal_k_elbow = k_range[0]
    
    # Find best silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    optimal_k_silhouette = k_range[best_sil_idx]
    
    # Create elbow plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k_elbow, color='red', linestyle='--', 
               label=f'Elbow method: k={optimal_k_elbow}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Silhouette plot
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k_silhouette, color='red', linestyle='--', 
               label=f'Best silhouette: k={optimal_k_silhouette}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'clustering_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering analysis saved to: {output_path}")
    print(f"Elbow method suggests: k={optimal_k_elbow}")
    print(f"Best silhouette score at: k={optimal_k_silhouette}")
    
    # Use the better of the two methods (or average if they're close)
    if abs(optimal_k_elbow - optimal_k_silhouette) <= 1:
        final_k = max(optimal_k_elbow, optimal_k_silhouette)
    else:
        # If they differ significantly, prefer silhouette score
        final_k = optimal_k_silhouette
    
    # Final clustering with chosen k
    kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(data_scaled)
    
    print(f"Final choice: k={final_k}")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")
    
    return cluster_labels, final_k, (inertias, silhouette_scores, k_range)


def create_pca_plot(data, cluster_labels, output_dir, n_components=2):
    """Create and save PCA visualization with cluster colors."""
    print("Computing PCA...")
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    
    # Create plot with cluster colors
    plt.figure(figsize=(12, 8))
    
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i, cluster_id in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=30)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'PCA Visualization with {n_clusters} Clusters\n'
              f'Total Explained Variance: {pca.explained_variance_ratio_.sum():.2%}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = Path(output_dir) / 'pca_clustered.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Clustered PCA plot saved to: {output_path}")
    
    return pca_result, pca.explained_variance_ratio_


def create_tsne_cluster_analysis(data, cluster_labels, output_dir, n_iter=1000):
    """Create t-SNE analysis with different perplexities and cluster consistency."""
    print("Computing t-SNE cluster analysis...")
    
    # Reduce dimensions with PCA first if data is high-dimensional
    if data.shape[1] > 50:
        pca = PCA(n_components=50)
        data_reduced = pca.fit_transform(data)
        print(f"Pre-reduced data from {data.shape[1]} to 50 dimensions with PCA")
    else:
        data_reduced = data
    
    # Calculate perplexity values based on dataset size
    n_samples = data_reduced.shape[0]
    perplexity_values = [
        max(5, min(10, n_samples // 10)),
        max(10, min(30, n_samples // 5)),
        max(20, min(50, n_samples // 3)),
        max(30, min(100, n_samples // 2))
    ]
    perplexity_values = sorted(list(set([p for p in perplexity_values if 5 <= p < n_samples])))
    
    print(f"Testing perplexity values: {perplexity_values}")
    
    # Create subplot grid
    n_plots = len(perplexity_values)
    cols = 2
    rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Colors for clusters
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    results = []
    cluster_preservation_scores = []
    
    for i, perp in enumerate(perplexity_values):
        print(f"Computing t-SNE with perplexity={perp}")
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter, 
                   random_state=42, verbose=0)
        tsne_result = tsne.fit_transform(data_reduced)
        results.append((perp, tsne_result))
        
        # Compute cluster preservation (how well t-SNE preserves original clusters)
        # Using k-means on t-SNE result and comparing with original clusters
        kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42)
        tsne_clusters = kmeans_tsne.fit_predict(tsne_result)
        preservation_score = adjusted_rand_score(cluster_labels, tsne_clusters)
        cluster_preservation_scores.append(preservation_score)
        
        # Plot with cluster colors
        if i < len(axes):
            for j, cluster_id in enumerate(np.unique(cluster_labels)):
                mask = cluster_labels == cluster_id
                axes[i].scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                               c=[colors[j]], alpha=0.7, s=20, label=f'C{cluster_id}' if i == 0 else "")
            
            axes[i].set_title(f't-SNE (perplexity={perp})\nCluster preservation: {preservation_score:.3f}')
            axes[i].set_xlabel('Component 1')
            axes[i].set_ylabel('Component 2')
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide unused subplots
    for i in range(len(perplexity_values), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'tsne_cluster_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE cluster analysis saved to: {output_path}")
    
    # Find best perplexity based on cluster preservation
    best_preservation_idx = np.argmax(cluster_preservation_scores)
    best_perplexity = perplexity_values[best_preservation_idx]
    best_tsne_result = results[best_preservation_idx][1]
    best_preservation = cluster_preservation_scores[best_preservation_idx]
    
    print(f"Best cluster preservation at perplexity={best_perplexity} (score: {best_preservation:.3f})")
    
    # Create individual best plot
    plt.figure(figsize=(12, 8))
    for j, cluster_id in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        plt.scatter(best_tsne_result[mask, 0], best_tsne_result[mask, 1], 
                   c=[colors[j]], alpha=0.7, s=30, label=f'Cluster {cluster_id}')
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f't-SNE Best Clustering (perplexity={best_perplexity})\n'
              f'Cluster Preservation Score: {best_preservation:.3f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'tsne_best_clustering.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best t-SNE clustering plot saved to: {output_path}")
    
    return best_tsne_result, best_perplexity, cluster_preservation_scores


def create_umap_cluster_analysis(data, cluster_labels, output_dir):
    """Create UMAP analysis with different parameters and cluster consistency."""
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping cluster analysis...")
        return None, None, []
        
    print("Computing UMAP cluster analysis...")
    
    # Parameter combinations to try
    n_neighbors_values = [5, 15, 30, 50]
    min_dist_values = [0.0, 0.1, 0.25, 0.5]
    
    # Filter based on dataset size
    n_samples = data.shape[0]
    n_neighbors_values = [n for n in n_neighbors_values if n < n_samples]
    
    # Try key combinations
    param_combinations = [
        (n_neighbors_values[0], min_dist_values[1]),  # Low neighbors, low dist
        (n_neighbors_values[1], min_dist_values[1]),  # Mid neighbors, low dist  
        (n_neighbors_values[1], min_dist_values[2]),  # Mid neighbors, mid dist
        (n_neighbors_values[-1], min_dist_values[2])  # High neighbors, mid dist
    ]
    
    # Colors for clusters
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    results = []
    cluster_preservation_scores = []
    
    for i, (n_neighbors, min_dist) in enumerate(param_combinations):
        print(f"Computing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                           n_components=2, random_state=42)
        umap_result = reducer.fit_transform(data)
        results.append((n_neighbors, min_dist, umap_result))
        
        # Compute cluster preservation
        kmeans_umap = KMeans(n_clusters=n_clusters, random_state=42)
        umap_clusters = kmeans_umap.fit_predict(umap_result)
        preservation_score = adjusted_rand_score(cluster_labels, umap_clusters)
        cluster_preservation_scores.append(preservation_score)
        
        # Plot with cluster colors
        for j, cluster_id in enumerate(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            axes[i].scatter(umap_result[mask, 0], umap_result[mask, 1], 
                           c=[colors[j]], alpha=0.7, s=20, label=f'C{cluster_id}' if i == 0 else "")
        
        axes[i].set_title(f'UMAP (n={n_neighbors}, d={min_dist})\nPreservation: {preservation_score:.3f}')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'umap_cluster_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UMAP cluster analysis saved to: {output_path}")
    
    # Find best parameters based on cluster preservation
    best_preservation_idx = np.argmax(cluster_preservation_scores)
    best_params = param_combinations[best_preservation_idx]
    best_umap_result = results[best_preservation_idx][2]
    best_preservation = cluster_preservation_scores[best_preservation_idx]
    
    print(f"Best UMAP cluster preservation: n_neighbors={best_params[0]}, min_dist={best_params[1]} (score: {best_preservation:.3f})")
    
    # Create individual best plot
    plt.figure(figsize=(12, 8))
    for j, cluster_id in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        plt.scatter(best_umap_result[mask, 0], best_umap_result[mask, 1], 
                   c=[colors[j]], alpha=0.7, s=30, label=f'Cluster {cluster_id}')
    
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title(f'UMAP Best Clustering (n_neighbors={best_params[0]}, min_dist={best_params[1]})\n'
              f'Cluster Preservation Score: {best_preservation:.3f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'umap_best_clustering.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best UMAP clustering plot saved to: {output_path}")
    
    return best_umap_result, best_params, cluster_preservation_scores


def create_final_comparison(pca_result, tsne_result, umap_result, cluster_labels, output_dir):
    """Create final comparison plot with best parameters and clustering."""
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # PCA
    for j, cluster_id in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=[colors[j]], alpha=0.7, s=30, label=f'Cluster {cluster_id}')
    axes[0].set_title('PCA with K-means Clusters')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # t-SNE
    for j, cluster_id in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       c=[colors[j]], alpha=0.7, s=30, label=f'Cluster {cluster_id}')
    axes[1].set_title('t-SNE with K-means Clusters')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].grid(True, alpha=0.3)
    
    # UMAP
    if umap_result is not None:
        for j, cluster_id in enumerate(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            axes[2].scatter(umap_result[mask, 0], umap_result[mask, 1], 
                           c=[colors[j]], alpha=0.7, s=30, label=f'Cluster {cluster_id}')
        axes[2].set_title('UMAP with K-means Clusters')
        axes[2].set_xlabel('Component 1')
        axes[2].set_ylabel('Component 2')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'UMAP\nNot Available', 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=16)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    
    plt.tight_layout()
    
    # Save comparison plot
    output_path = Path(output_dir) / 'final_clustered_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Final clustered comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing encoded .npy files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save visualization plots')
    parser.add_argument('--max-clusters', type=int, default=15,
                        help='Maximum number of clusters to test (default: 15)')
    parser.add_argument('--tsne-iter', type=int, default=1000,
                        help='Number of iterations for t-SNE (default: 1000)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.data_dir}")
    data = load_encoded_data(args.data_dir)
    print(f"Loaded data shape: {data.shape}")
    
    # Find optimal clusters
    print(f"\n{'='*60}")
    print("CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    
    cluster_labels, optimal_k, clustering_metrics = find_optimal_clusters(
        data, output_dir, max_clusters=args.max_clusters)
    
    # Create clustered visualizations
    print(f"\n{'='*60}")
    print("DIMENSIONALITY REDUCTION WITH CLUSTERING")
    print(f"{'='*60}")
    
    # PCA with clusters
    pca_result, explained_var = create_pca_plot(data, cluster_labels, output_dir)
    
    # t-SNE cluster analysis
    tsne_best, tsne_best_params, tsne_preservation = create_tsne_cluster_analysis(
        data, cluster_labels, output_dir, n_iter=args.tsne_iter)
    
    # UMAP cluster analysis
    umap_best, umap_best_params, umap_preservation = create_umap_cluster_analysis(
        data, cluster_labels, output_dir)
    
    # Final comparison
    create_final_comparison(pca_result, tsne_best, umap_best, cluster_labels, output_dir)
    
    # Save comprehensive analysis
    summary_file = output_dir / 'clustering_analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Clustering and Dimensionality Reduction Analysis\n")
        f.write(f"{'='*60}\n")
        f.write(f"Input directory: {args.data_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Feature dimensions: {data.shape[1]}\n")
        f.write(f"Number of samples: {data.shape[0]}\n\n")
        
        f.write(f"CLUSTERING RESULTS:\n")
        f.write(f"  Optimal number of clusters: {optimal_k}\n")
        f.write(f"  Cluster distribution: {np.bincount(cluster_labels)}\n")
        f.write(f"  Cluster percentages: {(np.bincount(cluster_labels) / len(cluster_labels) * 100).round(1)}%\n\n")
        
        f.write(f"PCA RESULTS:\n")
        f.write(f"  PC1 explained variance: {explained_var[0]:.4f}\n")
        f.write(f"  PC2 explained variance: {explained_var[1]:.4f}\n")
        f.write(f"  Total explained variance: {explained_var.sum():.4f}\n\n")
        
        f.write(f"t-SNE CLUSTER ANALYSIS:\n")
        f.write(f"  Best perplexity for cluster preservation: {tsne_best_params}\n")
        f.write(f"  Best cluster preservation score: {max(tsne_preservation):.3f}\n")
        f.write(f"  All preservation scores: {[f'{s:.3f}' for s in tsne_preservation]}\n\n")
        
        if umap_best_params:
            f.write(f"UMAP CLUSTER ANALYSIS:\n")
            f.write(f"  Best n_neighbors: {umap_best_params[0]}\n")
            f.write(f"  Best min_dist: {umap_best_params[1]}\n")
            f.write(f"  Best cluster preservation score: {max(umap_preservation):.3f}\n")
            f.write(f"  All preservation scores: {[f'{s:.3f}' for s in umap_preservation]}\n\n")
        
        f.write(f"RECOMMENDATIONS:\n")
        f.write(f"  Use {optimal_k} clusters for analysis\n")
        f.write(f"  Best t-SNE perplexity: {tsne_best_params}\n")
        if umap_best_params:
            f.write(f"  Best UMAP parameters: n_neighbors={umap_best_params[0]}, min_dist={umap_best_params[1]}\n")
        f.write(f"  Higher cluster preservation scores indicate better parameter choices\n")
    
    print(f"\n{'='*60}")
    print("Clustering analysis completed successfully!")
    print(f"Generated files:")
    print(f"  - Clustering analysis: clustering_analysis.png")
    print(f"  - PCA clustered: pca_clustered.png")
    print(f"  - t-SNE analysis: tsne_cluster_analysis.png")
    print(f"  - t-SNE best: tsne_best_clustering.png")
    if UMAP_AVAILABLE:
        print(f"  - UMAP analysis: umap_cluster_analysis.png")
        print(f"  - UMAP best: umap_best_clustering.png")
    print(f"  - Final comparison: final_clustered_comparison.png")
    print(f"  - Summary: clustering_analysis_summary.txt")
    print(f"\nKey Results:")
    print(f"  Optimal clusters: {optimal_k}")
    print(f"  Best t-SNE perplexity: {tsne_best_params}")
    if umap_best_params:
        print(f"  Best UMAP params: n_neighbors={umap_best_params[0]}, min_dist={umap_best_params[1]}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()