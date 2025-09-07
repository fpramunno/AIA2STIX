#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=performance
#SBATCH --job-name="train_visibility_refiner_composite"
#SBATCH --error=./logs/err/err_train_visibility_refiner_composite.log
#SBATCH --out=./logs/out/out_train_visibility_refiner_composite.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Refiner training parameters
config_file="/mnt/nas05/data01/francesco/AIA2STIX/training/configs/refiner_config_composite.yaml"

# Training parameters
epochs=1000                  # Number of epochs to train (overrides config file)
batch_size=32               # Batch size (overrides config file)

# Training data paths
train_diffusion_data="/mnt/nas05/data01/francesco/AIA2STIX/generated_visibilities/run_train_very_very_small_v2/generated_visibilities.npy"
train_true_data="/mnt/nas05/data01/francesco/AIA2STIX/generated_visibilities/run_train_very_very_small_v2/true_visibilities.npy"

# Validation data paths
val_diffusion_data="/mnt/nas05/data01/francesco/AIA2STIX/generated_visibilities/run_valid_very_very_small_v2/generated_visibilities.npy"
val_true_data="/mnt/nas05/data01/francesco/AIA2STIX/generated_visibilities/run_valid_very_very_small_v2/true_visibilities.npy"

checkpoint_dir="/mnt/nas05/data01/francesco/AIA2STIX/training/checkpoints/refiner_composite"
device="cuda"

# Weights & Biases configuration
use_wandb=true
wandb_project="aia2stix_refiner"
wandb_run_name="refiner_composite_mse_realimag"

python /mnt/nas05/data01/francesco/AIA2STIX/training/train_visibility_refiner.py \
    --config $config_file \
    --train-diffusion-data $train_diffusion_data \
    --train-true-data $train_true_data \
    --val-diffusion-data $val_diffusion_data \
    --val-true-data $val_true_data \
    --checkpoint-dir $checkpoint_dir \
    --device $device \
    --epochs $epochs \
    --batch-size $batch_size \
    --use-wandb $use_wandb \
    --wandb-project $wandb_project \
    --wandb-run-name $wandb_run_name