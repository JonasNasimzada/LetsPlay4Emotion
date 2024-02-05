#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --partition GPUampere
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
srun python ../utils/EMOCA/emoca_video_threading.py --input_directory "$1" --output_folder "$2"
