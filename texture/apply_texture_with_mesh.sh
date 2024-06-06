#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --cpus-per-task=64
srun python apply_uv_threading.py \
--input_directory "$1" \
--thread_num "$2"

