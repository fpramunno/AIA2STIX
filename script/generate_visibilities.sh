#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=performance
#SBATCH --job-name="generate_visibilities"
#SBATCH --error=./logs/err/err_generate_visibilities_valid.log
#SBATCH --out=./logs/out/out_generate_visibilities_valid.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE FOR YOUR SETUP
# =============================================================================

# Model and config paths
CONFIG_FILE="/mnt/nas05/data01/francesco/AIA2STIX/training/configs/very_very_small_model.json"
MODEL_PATH="/mnt/nas05/data01/francesco/AIA2STIX/saved_models/model_aia_2_stix_very_very_small_v2/model_epoch_0995.pth"

# Data paths
DATA_PATH="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images"
OUTPUT_DIR="/mnt/nas05/data01/francesco/AIA2STIX/generated_visibilities/run_valid_very_very_small_v2"

# Generation parameters
SAMPLER="dpmpp_2m"           # Options: euler, euler_ancestral, heun, dpm_2, dpm_2_ancestral, dpmpp_2m, dpmpp_2m_sde
NUM_STEPS=50                 # Number of diffusion steps (higher = better quality, slower)
NUM_SAMPLES_PER_IMAGE=1      # Number of visibility samples per AIA image
SPLIT="valid"                # Dataset split: train, val, test

# Processing parameters
BATCH_SIZE=16                # Batch size for processing
NUM_WORKERS=8                # Number of data loader workers
MAX_BATCHES=""               # Maximum batches to process (leave empty for all)

# System parameters
DEVICE="cuda"

# Build the command
python /mnt/nas05/data01/francesco/AIA2STIX/training/generate_visibilities.py \
    --config $CONFIG_FILE \
    --model-path $MODEL_PATH \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --sampler $SAMPLER \
    --num-steps $NUM_STEPS \
    --num-samples-per-image $NUM_SAMPLES_PER_IMAGE \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --split $SPLIT \
    --device $DEVICE
    