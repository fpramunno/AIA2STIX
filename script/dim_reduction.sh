#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="dim_reduction_aia_encoder"
#SBATCH --error=err_dim_reduction_aia_encoder.log
#SBATCH --out=out_dim_reduction_aia_encoder.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

data_path=/mnt/nas05/astrodata01/aia_2_stix/encoded_data/train/train
output_dir=/mnt/nas05/data01/francesco/AIA2STIX/clustering_analysis/


python /mnt/nas05/data01/francesco/AIA2STIX/analyze_encoded_features_clustering.py\
         --data-dir $data_path \
         --output-dir $output_dir \
         --max-clusters 10

