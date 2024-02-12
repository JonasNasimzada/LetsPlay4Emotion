import argparse
import os

import bpy

uv_material_list = [None, "Person_0"]
camera_position_list = ["Front", "Side"]


def load_and_render_mesh(file_path, output_dir):
    file_dir = os.path.basename(os.path.normpath(file_path))

    # importing the mesh group
    seq_imp_settings = bpy.types.PropertyGroup.bl_rna_get_subclass_py("SequenceImportSettings")
    seq_imp_settings.fileNamePrefix = bpy.props.StringProperty(name='File Name', default='0')
    bpy.ops.ms.import_sequence(directory=file_path)
    seq_imp_settings.fileNamePrefix = bpy.props.StringProperty(name='File Name')

    # adding uv texture to the mesh
    blend = bpy.data
    head = bpy.data.objects['000001_000_mesh_coarse_sequence']
    bpy.ops.ms.batch_shade_smooth()

    # render mesh for different uv-textures and camera-positions
    for uv_material in uv_material_list:
        if uv_material is not None:
            uv_texture = blend.materials[uv_material]
            head.active_material = uv_texture
        uv_material_dir = f"{output_dir}/{uv_material}"
        os.makedirs(uv_material_dir, exist_ok=True)
        for camera_position in camera_position_list:
            camera_position_dir = f"{uv_material_dir}/{camera_position}"
            os.makedirs(camera_position_dir, exist_ok=True)
            bpy.context.scene.camera = bpy.data.objects[camera_position]
            bpy.data.scenes[0].render.filepath = f"{camera_position_dir}/{file_dir}-{uv_material}-{uv_material}.mp4"
            bpy.ops.render.render(animation=True, write_still=True)
    bpy.ops.object.delete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    for dirs in os.listdir(args.input_dir):
        load_and_render_mesh(dirs, args.output_dir)
