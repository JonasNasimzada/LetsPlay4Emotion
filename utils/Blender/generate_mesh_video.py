import os
import shutil

import bpy

import argparse
import sys


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:


    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:

    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1:]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def load_and_render_mesh(input_path, file_path, output_path, uv_material_list, camera_position_list):
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
        if uv_material is None:
            uv_material = "Default OBJ"
        uv_texture = blend.materials[uv_material]
        head.active_material = uv_texture
        uv_material_dir = f"{output_path}/{uv_material}"
        os.makedirs(uv_material_dir, exist_ok=True)
        for camera_position in camera_position_list:
            camera_position_dir = f"{uv_material_dir}/{camera_position}"
            output_video = f"{file_dir}-{uv_material}-{camera_position}.mp4"
            output_path = f"{camera_position_dir}/{output_video}"
            os.makedirs(camera_position_dir, exist_ok=True)
            if os.path.isfile(output_video):
                continue
            bpy.data.scenes[0].render.filepath = f"/usr/local/work/{output_video}"
            bpy.context.scene.camera = bpy.data.objects[camera_position]
            bpy.ops.render.render(animation=True, write_still=True)
            shutil.move(f"/usr/local/work/{output_video}", output_path)

    bpy.ops.object.delete()


if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('--uv_material', nargs='+', default=[])
    parser.add_argument('--camera', nargs='+', default=[])
    parser.add_argument('--batch_files', nargs='+', default=[])

    args = parser.parse_args()
    # enable_gpus("CUDA")
    input_dir = "/usr/local/videos_input"
    output_dir = "/usr/local/videos_output"
    for dir_name in args.batch_files:
        load_and_render_mesh(input_dir, dir_name, output_dir, args.uv_material, args.camera)
