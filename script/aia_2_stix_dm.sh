#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_aia_2_stix_aug_35"
#SBATCH --error=./logs/err/err_training_aia_2_stix_aug_35.log
#SBATCH --out=./logs/out/out_training_aia_2_stix_aug_35.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

epochs=1000
batch_size=32
config_file="/mnt/nas05/data01/francesco/AIA2STIX/training/configs/3dmag.json"
saving_path="/mnt/nas05/data01/francesco/AIA2STIX/saved_models/"
dir_name="aia_2_stix_aug_35"
wandb_run_name="aia_2_stix_aug_35"

# Dataset paths
data_path_encoded="/mnt/nas05/astrodata01/aia_2_stix/encoded_data_augmented_35aug/"
train_data_path="/mnt/nas05/astrodata01/aia_2_stix/augmented_data_35aug_per_sample/"
val_data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images"
val_vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv"
val_enc_data_path="/mnt/nas05/astrodata01/aia_2_stix/encoded_data_valid/"

# Training configuration
save_every=5
evaluate_every=5

# Date ranges for data filtering
train_start_date="210522"
train_end_date="241122"
val_start_date="241122"
val_end_date="241230"

python /mnt/nas05/data01/francesco/AIA2STIX/training/train_3dmag.py --max-epochs $epochs \
    --batch-size $batch_size \
    --config $config_file \
    --saving-path $saving_path \
    --dir-name $dir_name \
    --wandb-run-name $wandb_run_name \
    --data-path-encoded $data_path_encoded \
    --train-data-path $train_data_path \
    --train-use-augmented \
    --val-data-path $val_data_path \
    --val-vis-path $val_vis_path \
    --val-enc-data-path $val_enc_data_path \
    --save-every $save_every \
    --evaluate-every $evaluate_every \
    --date-ranges train $train_start_date $train_end_date valid $val_start_date $val_end_date
