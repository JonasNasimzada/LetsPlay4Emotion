import os
import random
import shutil
import time
import argparse
import sys

import bpy


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    Custom ArgumentParser to handle Blender's '--' argument separator.
    """

    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1:]  # the list after '--'
        except ValueError:
            return []

    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())


def load_and_render_mesh(input_path, file_path, output_path, uv_material_list, camera_position_list):
    """
    Load and render the mesh with specified UV materials and camera positions.
    """
    file_dir = os.path.basename(os.path.normpath(file_path))
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 1

    # Importing the mesh group
    seq_imp_settings = bpy.types.PropertyGroup.bl_rna_get_subclass_py("SequenceImportSettings")
    seq_imp_settings.fileNamePrefix = bpy.props.StringProperty(name='File Name', default='0')
    bpy.ops.ms.import_sequence(directory=f"{input_path}/{file_path}")
    seq_imp_settings.fileNamePrefix = bpy.props.StringProperty(name='File Name')

    # Adding UV texture to the mesh
    blend = bpy.data
    head = bpy.data.objects['000001_000_mesh_coarse_sequence']
    bpy.ops.ms.batch_shade_smooth()

    is_random = False
    if not uv_material_list:
        is_random = True
        uv_material_list.append(random.choice(os.listdir("/usr/local/ffhq-textures")))

    # Render mesh for different UV textures and camera positions
    for uv_material in uv_material_list:
        uv_dir = ""
        if is_random:
            uv_dir = "random"
            uv_texture = bpy.data.materials["Person_0"]
            load_uv_texture(uv_texture, uv_material)
            uv_material_list.remove(uv_material)
        elif uv_material == "Default":
            uv_dir = uv_material
            uv_texture = bpy.data.materials["Default OBJ"]
        else:
            uv_dir = uv_material
            uv_texture = bpy.data.materials["Person_0"]
            load_uv_texture(uv_texture, uv_material)

        head.active_material = uv_texture
        for camera_position in camera_position_list:
            camera_position_dir = f"{output_path}/{uv_dir}/{camera_position}"
            output_video = f"{file_dir}-{uv_dir}-{camera_position}.mp4"
            output_path = f"{camera_position_dir}/{output_video}"
            os.makedirs(camera_position_dir, exist_ok=True)
            if os.path.isfile(output_video):
                bpy.ops.object.delete()
                continue
            bpy.data.scenes[0].render.filepath = f"/usr/local/work/{output_video}"
            bpy.context.scene.camera = bpy.data.objects[camera_position]
            bpy.ops.render.render(animation=True, write_still=True)
            shutil.move(f"/usr/local/work/{output_video}", output_path)

    bpy.ops.object.delete()


def load_uv_texture(uv_texture, uv_material):
    tree = uv_texture.node_tree
    nodes = tree.nodes
    node = nodes["Image Texture"]
    imgs = bpy.data.images
    img = imgs.load(f"/usr/local/ffhq-textures/{uv_material}")
    node.image = img


if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('--uv_material', nargs='+', default=[])
    parser.add_argument('--camera', nargs='+', default=[])
    parser.add_argument('--batch_files', nargs='+', default=[])

    args = parser.parse_args()
    input_dir = "/usr/local/videos_input"
    output_dir = "/usr/local/videos_output"
    tic = time.time()
    for dir_name in args.batch_files:
        load_and_render_mesh(input_dir, dir_name, output_dir, args.uv_material, args.camera)
    toc = time.time()
    print(f'meshes done, took {toc - tic} seconds.')
