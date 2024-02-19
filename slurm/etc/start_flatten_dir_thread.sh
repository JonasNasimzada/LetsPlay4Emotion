#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --cpus-per-gpu=64
srun python ../utils/flatten_dir_threading \
--input_directory /groups/constantin_students/jnasimzada/all_videos/ \
--output_dir /groups/constantin_students/jnasimzada/all_mesh/ \
--thread_num 64

