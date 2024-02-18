#!/usr/bin/env bash
#SBATCH --time 1-1
#SBATCH --partition GPUampere
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
srun python generating_threading.py \
--blend_file mesh_sequence_v3.blend \
--thread_num 85 \
--batch_size 4 \
--current_batch 3
