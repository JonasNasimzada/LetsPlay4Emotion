#!/usr/bin/env bash
#SBATCH --time 6-1
#SBATCH --partition GPUampere
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
srun python ../model/model_v2.py --mode train --type multi --version 1 --devices 2