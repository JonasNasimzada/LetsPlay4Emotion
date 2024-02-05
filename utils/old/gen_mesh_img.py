import os
import sys
import argparse
import torch

import pytorch3d
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.structures.meshes import (
    join_meshes_as_batch,
    join_meshes_as_scene,
    Meshes,
)


def create_mesh_img(dirname, input, out_dir):
    obj_file_head = f"{input}/{dirname}/stage3_mesh_exp.obj"
    obj_file_eye_l = f"{input}/{dirname}/L_ball.obj"
    obj_file_eye_r = f"{input}/{dirname}/R_ball.obj"
    print(f"working on file: {obj_file_head}")
    mesh_head = load_objs_as_meshes([obj_file_head], device=device)
    mesh_eye_l = load_objs_as_meshes([obj_file_eye_l], device=device)
    mesh_eye_r = load_objs_as_meshes([obj_file_eye_r], device=device)
    meshes = join_meshes_as_scene([mesh_head, mesh_eye_l, mesh_eye_r])
    R, T = look_at_view_transform(dist=2.7, elev=10, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=3000,
        blur_radius=0.0,
        faces_per_pixel=30,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
        )
    )
    images = renderer(meshes).cpu().numpy()
    plt.imsave(f"{out_dir}/{dirname}.png", images[0, ..., :3].cpu().numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dir_mesh', type=str, required=True)
    parser.add_argument('--output_dir_mesh_img', type=str, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    for root, dirs, files in os.walk(args.dir_mesh):
        if not dirs:
            working_dir = os.path.relpath(root, args.dir_mesh)
            create_mesh_img(working_dir, root, args.output_dir_mesh_img)
