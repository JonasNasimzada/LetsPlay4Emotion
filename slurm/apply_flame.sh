#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --cpus-per-gpu=64
srun python ../FFHQ-UV/run_flame_apply_hifi3d_uv.py \
--input_directory /groups/constantin_students/jnasimzada/all_videos/ \
--output_dir /groups/constantin_students/jnasimzada/all_mesh/ \
--thread_num 64

