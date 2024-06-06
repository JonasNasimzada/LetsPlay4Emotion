#!/usr/bin/env bash
#SBATCH --partition GPUampere
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_dir) input_dir="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        *) usage ;;
    esac
    shift
done
srun ./FFHQ-UV/run_rgb_fitting_realy.sh "$input_dir" "$output_dir"
