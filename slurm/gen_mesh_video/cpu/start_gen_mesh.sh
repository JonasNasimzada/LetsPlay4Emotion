#!/bin/bash

n=30
for ((i = 1; i <= n; i++)); do
    echo "Processing number $i"
    sbatch --time 2-1 --cpus-per-task=64 --wrap="python generating_threading.py \
    --blend_file mesh_sequence_v3.blend \
    --thread_num 16 \
    --batch_size $n \
    --current_batch $((i-1))"
done