import os
import pathlib
import sys
import argparse
import torch

sys.path.append("./FFHQ-UV/RGB_Fitting/")
sys.path.append("./FFHQ-UV/Mesh_Add_EyeBall/")
sys.path.append("./FFHQ-UV/RGB_Fitting/utils")
sys.path.append("./FFHQ-UV/RGB_Fitting/model")

import step2_fit_processed_data
import step1_process_data
from utils.data_utils import setup_seed
import run_mesh_add_eyeball

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='pretrained models.')
    parser.add_argument('--topo_dir', type=str, default='../topo_assets', help='assets of topo.')
    parser.add_argument('--texgan_model_name', type=str, default='texgan_ffhq_uv.pth', help='texgan model name.')
    parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    parser.add_argument('--input_gen_mesh', type=str, required=True)
    parser.add_argument('--output_dir_mesh', type=str, required=True)
    args = parser.parse_args()
    setup_seed(777)
    for root, dirs, files in os.walk(args.input_gen_mesh):
        if not dirs:
            working_dir = os.path.relpath(root, pathlib.Path().resolve())
            print(f"for dir {working_dir} meshes will be generated")
            output_dir = f"{args.output_dir_mesh}/{'/'.join(root.split('/')[1:])}"
            print(f"output dir for mesh is:  {output_dir}")
            step1_process_data.process_data(args, working_dir, f"{working_dir}/processed_data")
            step2_fit_processed_data.generate_mesh(args, f"{working_dir}/processed_data", output_dir)

    for root, dirs, files in os.walk(args.output_dir_mesh):
        if not dirs:
            working_dir = os.path.relpath(root, pathlib.Path().resolve())
            run_mesh_add_eyeball.start_add_mesh(working_dir)
            print(f"finished with generating dir {working_dir}")
