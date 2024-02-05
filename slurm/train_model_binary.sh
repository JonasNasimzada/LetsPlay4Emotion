#!/usr/bin/env bash
#SBATCH --time 6-1
#SBATCH --partition GPUampere
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=4
python ../model/model_v2.py --mode train --type binary --version 1 --devices 3
