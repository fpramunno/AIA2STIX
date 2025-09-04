#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_aia_2_stix_encoder"
#SBATCH --error=./logs/err/err_training_aia_2_stix_encoder.log
#SBATCH --out=./logs/out/out_training_aia_2_stix_encoder.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

epochs=1000
batch_size=32
encoder_checkpoint="/mnt/nas05/data01/francesco/AIA2STIX/encoder_decoder_checkpoints_palette/checkpoint_epoch_115.pth"
wandb_run_name="encoder_frozen_v1"
data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images"
vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv"
saving_path="/mnt/nas05/data01/francesco/AIA2STIX/saved_models/"
dir_name="encoder_to_vis_v1"

python /mnt/nas05/data01/francesco/AIA2STIX/training/train_encoder_to_visibility.py \
      --encoder-checkpoint $encoder_checkpoint \
      --data-path $data_path \
      --vis-path $vis_path \
      --freeze-encoder \
      --max-epochs $epochs \
      --wandb-run-name $wandb_run_name \
      --saving-path $saving_path \
      --dir-name $dir_name \
      --use-wandb

