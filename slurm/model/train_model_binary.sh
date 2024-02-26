#!/usr/bin/env bash
#SBATCH --time 6-1
#SBATCH --partition GPUampere
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
#SBATCH --nodes=1
srun python model_v2.py --mode train --type binary --version 3 --devices 2 --logger_comment Person_1_Front_augmen_adjust --dataset_path datasets/Person_1_Front
