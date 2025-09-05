#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="enc_data_val"
#SBATCH --error=./logs/err/err_enc_data_val.log
#SBATCH --out=./logs/out/out_enc_enc_data_val.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

ckpt_path=/mnt/nas05/data01/francesco/AIA2STIX/encoder_decoder_checkpoints_palette/checkpoint_epoch_115.pth
data_path=/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images
output_dir=/mnt/nas05/astrodata01/aia_2_stix/encoded_data_valid/
train_start=241122
train_end=241230

python training/encode_model_bot3.py \
      --checkpoint-path $ckpt_path \
      --data-path $data_path \
      --output-dir $output_dir \
      --encode-splits valid \
      --date-ranges valid $train_start $train_end



