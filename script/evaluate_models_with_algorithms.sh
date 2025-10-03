#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="evaluate_aia_2_stix_algorithms"
#SBATCH --error=./logs/err/err_evaluate_aia_2_stix_algorithms.log
#SBATCH --out=./logs/out/out_evaluate_aia_2_stix_algorithms.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Model Configuration
model_type="diffusion"  # Options: "diffusion" or "encoder"
model_checkpoint="/mnt/nas05/data01/francesco/AIA2STIX/saved_models/model_aia_2_stix_very_very_small_v2/model_epoch_2990.pth"
encoder_checkpoint="/mnt/nas05/data01/francesco/AIA2STIX/encoder_decoder_checkpoints_palette/checkpoint_epoch_115.pth"
config_file="/mnt/nas05/data01/francesco/AIA2STIX/training/configs/very_very_small_model.json"

# Algorithm Selection - Choose which reconstruction algorithms to use
# Available options: fcd, clean, mem, em, back_projection
# Examples:
#   algorithms="fcd"                           # Only FCD (original behavior)
#   algorithms="fcd clean mem"                 # FCD + selected STIX algorithms
#   algorithms="clean mem em back_projection"  # All STIX algorithms
#   algorithms="fcd clean mem em back_projection"  # All algorithms
algorithms="fcd clean mem em"  # Default: FCD + 3 STIX algorithms

# FCD Model Configuration
fcd_model_path=""  # Leave empty to download from HuggingFace
fcd_backend="tensorflow"  # Options: "jax", "torch", "tensorflow"
fcd_download_dir="/mnt/nas05/data01/francesco/AIA2STIX/fcd_models/"  # Directory to download FCD model

# Data Paths
data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images"
vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv"
enc_data_path="/mnt/nas05/astrodata01/aia_2_stix/encoded_data/"  # For diffusion model conditioning

# Evaluation Settings
batch_size=16
split="valid"  # Options: "train", "valid", "test"

# Create output directory with algorithm names
algorithms_suffix=$(echo $algorithms | tr ' ' '_')
output_dir="/mnt/nas05/data01/francesco/AIA2STIX/evaluation_results/${model_type}_$(basename $model_checkpoint .pth)_${algorithms_suffix}_${split}"

# Create directories
mkdir -p "$output_dir"
mkdir -p "./logs/err"
mkdir -p "./logs/out"
if [ -n "$fcd_download_dir" ]; then
    mkdir -p "$fcd_download_dir"
fi

# Print configuration
echo "==================================="
echo "AIA2STIX Multi-Algorithm Evaluation"
echo "==================================="
echo "Model type: $model_type"
echo "Model checkpoint: $(basename $model_checkpoint)"
echo "Algorithms: $algorithms"
echo "Output directory: $output_dir"
echo "==================================="

# Run evaluation with algorithm selection
common_args="--model-type $model_type \
    --model-checkpoint $model_checkpoint \
    --data-path $data_path \
    --vis-path $vis_path \
    --batch-size $batch_size \
    --split $split \
    --output-dir $output_dir \
    --algorithms $algorithms"

# Add FCD-specific arguments if FCD is selected
if [[ "$algorithms" == *"fcd"* ]]; then
    fcd_args="--fcd-backend $fcd_backend"
    if [ -n "$fcd_download_dir" ]; then
        fcd_args="$fcd_args --fcd-download-dir $fcd_download_dir"
    fi
    if [ -n "$fcd_model_path" ]; then
        fcd_args="$fcd_args --fcd-model-path $fcd_model_path"
    fi
    common_args="$common_args $fcd_args"
fi

# Model-specific arguments and execution
if [ "$model_type" == "diffusion" ]; then
    echo "Running diffusion model evaluation..."
    python /mnt/nas05/data01/francesco/AIA2STIX/training/evaluate_models_with_algorithms.py \
        $common_args \
        --config $config_file \
        --enc-data-path $enc_data_path

elif [ "$model_type" == "encoder" ]; then
    echo "Running encoder model evaluation..."
    python /mnt/nas05/data01/francesco/AIA2STIX/training/evaluate_models_with_algorithms.py \
        $common_args \
        --encoder-checkpoint $encoder_checkpoint

else
    echo "Error: Invalid model_type '$model_type'. Must be 'diffusion' or 'encoder'."
    exit 1
fi

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: $output_dir"
    echo "==================================="
    
    # List generated files
    echo "Generated files:"
    ls -la "$output_dir"/*.png 2>/dev/null || echo "No PNG files found"
    ls -la "$output_dir"/*.npz 2>/dev/null || echo "No NPZ files found"
else
    echo "==================================="
    echo "Evaluation failed! Check logs for details."
    echo "Error log: ./logs/err/err_evaluate_aia_2_stix_algorithms.log"
    echo "==================================="
    exit 1
fi