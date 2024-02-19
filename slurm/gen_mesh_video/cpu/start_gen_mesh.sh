#!/bin/bash

for ((i = 1; i <= 20; i++)); do
    echo "Processing number $i"
    sbatch --time 2-1 --cpus-per-task=64 --wrap="python generating_threading.py \
    --blend_file mesh_sequence_v3.blend \
    --thread_num 16 \
    --batch_size 20 \
    --current_batch $i"
done