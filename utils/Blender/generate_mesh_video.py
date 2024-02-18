import argparse
import os
from custom_argument_parser import ArgumentParserForBlender
import bpy

uv_material_list = ["Person_0"]
camera_position_list = ["Front", "Side"]
rendered_video = []


def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


enable_gpus("CUDA")


def load_and_render_mesh(input_path, file_path, output_path):
    file_dir = os.path.basename(os.path.normpath(file_path))
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 1

    # importing the mesh group
    seq_imp_settings = bpy.types.PropertyGroup.bl_rna_get_subclass_py("SequenceImportSettings")
    seq_imp_settings.fileNamePrefix = bpy.props.StringProperty(name='File Name', default='0')
    bpy.ops.ms.import_sequence(directory=f"{input_path}/{file_path}")
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
        uv_material_dir = f"{output_path}/{uv_material}"
        os.makedirs(uv_material_dir, exist_ok=True)
        for camera_position in camera_position_list:
            camera_position_dir = f"{uv_material_dir}/{camera_position}"
            os.makedirs(camera_position_dir, exist_ok=True)
            bpy.context.scene.camera = bpy.data.objects[camera_position]
            bpy.data.scenes[0].render.filepath = f"{camera_position_dir}/{file_dir}-{uv_material}-{camera_position}.mp4"
            bpy.ops.render.render(animation=True, write_still=True)
            print(f"generated video: {file_dir} - {camera_position}")
    bpy.ops.object.delete()
    rendered_video.append(file_dir)
    print("all generated files:")
    print(file_dir)


if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('--batch_files', nargs='+', default=[])

    args = parser.parse_args()
    # enable_gpus("CUDA")
    input_dir = "/usr/local/videos_input"
    output_dir = "/usr/local/videos_output"
    for dir_name in args.batch_files:
        load_and_render_mesh(input_dir, dir_name, output_dir)
