#!/bin/bash

echo "Enter list of uv-textures (Default, Person_0,...) (separated by spaces):"
read -r uv_material

echo "Enter the camera views (Front or Side) (separated by spaces):"
read -r camera

echo "Enter blend file version:"
read -r blend_file_version

echo "Enter amount of nodes:"
read -r number

echo "Enter a job dependency (if any) or leave blank:"
read -r dependency


IFS=' ' read -r -a uv_material_list <<< "$uv_material"
IFS=' ' read -r -a camera_list <<< "$camera"

echo "Processing uv-textures: ${uv_material_list[*]}"
echo "Processing camera views: ${camera_list[*]}"
echo "Processing amount of nodes: $number"
if [[ -n "$dependency" ]]; then
        echo "Processing dependency: $dependency"
    else
        echo "No dependency entered"
    fi
echo ""

mamba activate py3.9v2

for ((i = 1; i <= number; i++)); do
    sbatch_command="sbatch --time 1-1 --cpus-per-task=64"
    if [[ -n "$dependency" ]]; then
        sbatch_command+=" --dependency afterok:$dependency"
    fi
    sbatch_command+=" --wrap=\"python generating_threading.py \
        --blend_file mesh_sequence_v$blend_file_version.blend \
        --thread_num 16 \
        --batch_size $number \
        --current_batch $((i-1))"
    if [[ -n "${uv_material_list[*]}" ]]; then
        sbatch_command+=" --uv_material ${uv_material_list[*]}"
    fi
    if [[ -n "${camera_list[*]}" ]]; then
        sbatch_command+=" --camera ${camera_list[*]}"
    fi
    sbatch_command+="\""
    eval "$sbatch_command"
done

echo ""
sleep 1
squeue -u jnasimzada