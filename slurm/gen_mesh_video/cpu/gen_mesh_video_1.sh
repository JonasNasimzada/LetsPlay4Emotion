#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --cpus-per-task=64
srun python generating_threading.py \
--blend_file mesh_sequence_v3.blend \
--thread_num 16 \
--batch_size 15 \
--current_batch 1
