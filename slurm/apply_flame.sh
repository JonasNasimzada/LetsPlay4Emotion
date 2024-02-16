#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --cpus-per-task=64
srun python apply_uv_threading.py \
--input_directory /homes/jnasimzada/test_mest \
--thread_num 1

