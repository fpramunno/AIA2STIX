#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="evaluate_aia_2_stix_models"
#SBATCH --error=./logs/err/err_evaluate_aia_2_stix_models.log
#SBATCH --out=./logs/out/out_evaluate_aia_2_stix_models.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Model Configuration
model_type="diffusion"  # Options: "diffusion" or "encoder"
model_checkpoint="/mnt/nas05/data01/francesco/AIA2STIX/model_aia_2_stix_v1/model_00002800.pth"
encoder_checkpoint="/mnt/nas05/data01/francesco/AIA2STIX/encoder_decoder_checkpoints_palette/checkpoint_epoch_115.pth"
config_file="/mnt/nas05/data01/francesco/AIA2STIX/training/configs/3dmag.json"

# FCD Model
fcd_model_path=""  # Leave empty to download from HuggingFace
fcd_backend="tensorflow"  # Options: "jax", "torch", "tensorflow"
fcd_download_dir="/mnt/nas05/data01/francesco/AIA2STIX/fcd_models/"  # Directory to download FCD model

# Data Paths
data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images"
vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv"
enc_data_path="/mnt/nas05/astrodata01/aia_2_stix/encoded_data/"  # For diffusion model conditioning

# Evaluation Settings
batch_size=16
split="train"  # Options: "train", "valid", "test"
num_batches=20
output_dir="/mnt/nas05/data01/francesco/AIA2STIX/evaluation_results/${model_type}_checkpoint_$(basename $model_checkpoint .pth)_on_${split}"

# Create directories
mkdir -p "$output_dir"
mkdir -p $fcd_download_dir

# Run evaluation
if [ "$model_type" == "diffusion" ]; then
    python /mnt/nas05/data01/francesco/AIA2STIX/training/evaluate_models.py \
        --model-type $model_type \
        --model-checkpoint $model_checkpoint \
        --config $config_file \
        --data-path $data_path \
        --vis-path $vis_path \
        --enc-data-path $enc_data_path \
        --batch-size $batch_size \
        --split $split \
        --num-batches $num_batches \
        --output-dir $output_dir \
        --fcd-backend $fcd_backend \
        --fcd-download-dir $fcd_download_dir \
        ${fcd_model_path:+--fcd-model-path "$fcd_model_path"}

elif [ "$model_type" == "encoder" ]; then
    python /mnt/nas05/data01/francesco/AIA2STIX/training/evaluate_models.py \
        --model-type $model_type \
        --model-checkpoint $model_checkpoint \
        --encoder-checkpoint $encoder_checkpoint \
        --data-path $data_path \
        --vis-path $vis_path \
        --batch-size $batch_size \
        --split $split \
        --num-batches $num_batches \
        --output-dir $output_dir \
        --fcd-backend $fcd_backend \
        --fcd-download-dir $fcd_download_dir \
        ${fcd_model_path:+--fcd-model-path "$fcd_model_path"}
fi