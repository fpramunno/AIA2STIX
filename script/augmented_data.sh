#!/bin/bash
#SBATCH --gres=gpu:0  
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="augmented_dataset_3"
#SBATCH --error=./logs/err/augmented_dataset_3.log
#SBATCH --out=./logs/out/augmented_dataset_3.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

output_dir='/mnt/nas05/astrodata01/aia_2_stix/augmented_data_35aug_per_sample'
aug_per_sample=35
rotation_min=-90
rotation_max=90

python /mnt/nas05/data01/francesco/AIA2STIX/training/generate_augmented_dataset.py --full-dataset --output-dir $output_dir --aug-per-sample $aug_per_sample --rotation-range $rotation_min $rotation_max