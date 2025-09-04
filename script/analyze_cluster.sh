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
output_dir=/mnt/nas05/data01/francesco/AIA2STIX/clustering_analysis/check_cluster_characteristics
original_crop=/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images
csv_file=/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv


python /mnt/nas05/data01/francesco/AIA2STIX/analyze_cluster_characteristics.py\
         --encoded-dir $data_path \
         --processed-images-dir $original_crop \
         --csv-file $csv_file \
         --output-dir $output_dir \
         --n-clusters 2
