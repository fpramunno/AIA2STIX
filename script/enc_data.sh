#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_aia_encoder"
#SBATCH --error=err_training_aia_encoder.log
#SBATCH --out=out_training_aia_encoder.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

ckpt_path=/mnt/nas05/data01/francesco/AIA2STIX/encoder_decoder_checkpoints_palette/checkpoint_epoch_115.pth
output_dir=/mnt/nas05/astrodata01/aia_2_stix/encoded_data/


python /mnt/nas05/data01/francesco/AIA2STIX/training/encode_model_bot3.py \
         --checkpoint-path $ckpt_path \
         --output-dir $output_dir \

