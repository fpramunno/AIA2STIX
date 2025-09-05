#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --partition=performance
#SBATCH --job-name="training_aia_2_stix_veryverysmall"
#SBATCH --error=./logs/err/err_training_aia_2_stix_veryverysmall.log
#SBATCH --out=./logs/out/out_training_aia_2_stix_veryverysmall.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

epochs=1000
batch_size=32
evaluate_every=5
save_every=5
config_file="/mnt/nas05/data01/francesco/AIA2STIX/training/configs/very_very_small_model.json"
saving_path="/mnt/nas05/data01/francesco/AIA2STIX/saved_models/"
dir_name="aia_2_stix_very_very_small"
wandb_run_name="aia_2_stix_very_very_small"

python /mnt/nas05/data01/francesco/AIA2STIX/training/train_3dmag.py --max-epochs $epochs \
    --batch-size $batch_size \
    --evaluate-every $evaluate_every \
    --save-every $save_every \
    --config $config_file \
    --saving-path $saving_path \
    --dir-name $dir_name \
    --wandb-run-name $wandb_run_name

