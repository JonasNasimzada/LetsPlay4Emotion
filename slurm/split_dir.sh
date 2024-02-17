#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --cpus-per-task=64
srun python split_dir.py \
--input_directory /groups/constantin_students/jnasimzada/all_mesh/ \
--dirs 4

