#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --partition GPUampere
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
srun python create_video_threading.py \
--input_directory /groups/constantin_students/jnasimzada/output_mesh \
--output_dir /groups/constantin_students/jnasimzada/video_head \
--thread_num 100

