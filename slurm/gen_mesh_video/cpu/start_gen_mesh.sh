#!/bin/bash

echo "Enter list of uv_materials (None, Person_1,...) (separated by spaces):"
read -r uv_material

echo "Enter the camera view (Front or Side) (separated by spaces):"
read -r camera

echo "Enter amount of nodes:"
read -r number

echo "Enter a job dependency (if any) or leave blank:"
read -r dependency


IFS=' ' read -r -a uv_material_list <<< "$uv_material"
IFS=' ' read -r -a camera_list <<< "$camera"


for ((i = 1; i <= number; i++)); do
    echo "Processing number $i"
    sbatch_command="sbatch --time 1-1 --cpus-per-task=64"
    if [[ -n "$dependency" ]]; then
        sbatch_command+=" --dependency=$dependency"
    fi
    sbatch_command+=" --wrap=\"python generating_threading.py \
        --blend_file mesh_sequence_v3.blend \
        --thread_num 16 \
        --batch_size $number \
        --current_batch $((i-1)) \
        --uv_material ${uv_material_list[*]} \
        --camera ${camera_list[*]}\""
    eval "$sbatch_command"
done