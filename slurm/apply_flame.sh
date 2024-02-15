#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --cpus-per-task=64
srun python ../FFHQ-UV/apply_uv_threading.py \
--input_directory /groups/constantin_students/jnasimzada/all_videos/ \
--thread_num 1

