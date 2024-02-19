#!/usr/bin/env sh
#SBATCH --time=1-1
#SBATCH --partition GPUampere
#SBATCH --gpus=1

srun python ./gen_mesh.py \
--texgan_model_name texgan_ffhq_uv.pth \
--checkpoints_dir FFHQ-UV/checkpoints \
--topo_dir FFHQ-UV/topo_assets \
--input_gen_mesh test_input_video_v2/out_frames \
--output_dir_mesh output_3d_mesh/flatten_dir_EMOCA_v1
