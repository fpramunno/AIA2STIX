#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodelist=server0094
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_aia_encoder"
#SBATCH --error=err_training_aia_encoder.log
#SBATCH --out=out_training_aia_encoder.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G


python /mnt/nas05/data01/francesco/AIA2STIX/training/train_encoder_decoder.py --wandb-project "aia_encoder" --wandb-name "run-1" --save-dir "./encoder_decoder_checkpoints_palette" --epochs 500 

