#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition GPUampere
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8

srun python model_v5.py \
--mode train \
--type binary \
--version 45 \
--devices 1 \
--train_dataset <train_dataset> \
--val_dataset <val_dataset> \
--logger_comment <logger_comment> \
--batch_size 16 \
--epochs 200 \
--video_path_prefix <video_path_prefix>