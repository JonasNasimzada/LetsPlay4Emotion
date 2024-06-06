import argparse
import os
import shutil
import subprocess
import time
import torch.multiprocessing as mp

forbidden_folder = ".ipynb_checkpoints"


def split_list_into_parts(input_list, n, y):
    """
    Splits the input list into 'n' parts and returns the 'y'-th part.
    """
    part_length = len(input_list) // n
    parts = []
    for i in range(n):
        if i > y:
            continue
        start_index = i * part_length
        end_index = (i + 1) * part_length if i < n - 1 else None
        current_part = input_list[start_index:end_index]
        parts.append(current_part)

    return parts[y] if 0 <= y < n else None


class BlenderProcess(mp.Process):
    """
    Multiprocessing class to run Blender command in separate processes.
    """

    def __init__(self, process_number, directories, blender_file, uv_texture, camera, location):
        super().__init__()
        self.process_number = process_number
        self.directories = directories
        self.blender_file = blender_file
        self.uv_texture = uv_texture
        self.camera = camera
        self.location = location

    def run(self):
        """
        Override the run method to execute the Blender command.
        """
        print(f"Process {self.process_number} is running")
        if forbidden_folder in self.directories:
            self.directories.remove(forbidden_folder)
        self.execute_command()

    def execute_command(self):
        """
        Constructs and runs the Blender command.
        """
        command = (
            f"apptainer exec "
            f"--bind ${self.location}/blender/Stop-motion-OBJ:/usr/local/blender/3.6/scripts/addons/Stop-motion-OBJ "
            f"--bind ${self.location}/blender/config:/usr/local/blender/3.6/config/ "
            f"--bind ${self.location}/all_mesh:/usr/local/videos_input "
            f"--bind ${self.location}/videos_mesh:/usr/local/videos_output "
            f"--bind ${self.location}/render:/usr/local/render_scripts_scripts "
            f"--bind ${self.location}/blender:/usr/local/blend_file "
            f"--bind /local/work/:/usr/local/work/ "
            f"--bind ${self.location}/ffhq_textures:/usr/local/work/ffhq-textures "
            f"--nv docker://blendergrid/blender:3.6.8 "
            f"/usr/local/blender/blender "
            f"--background "
            f"/usr/local/blend_file/{self.blender_file} "
            f"--python /usr/local/render_scripts/generate_mesh_video.py "
            f"--"
        )
        if self.uv_texture:
            command += f" --uv_material {' '.join(self.uv_texture)}"
        command += (
            f" --camera {' '.join(self.camera)} "
            f"--batch_files {' '.join(self.directories)}"
        )
        print(command)
        try:
            subprocess.run(command, capture_output=True, shell=True, check=True)
            print(f"successful run for dirs: {' '.join(self.directories)}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command in Process {self.process_number} for directory {self.directories}: {e}")


def main():
    """
    Main function to parse arguments and start multiprocessing Blender processes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend_file', type=str, required=True, help="Path to the Blender file.")
    parser.add_argument('--thread_num', type=int, default=1, help="Number of threads/processes.")
    parser.add_argument('--batch_size', type=int, default=1, help="Total number of batches.")
    parser.add_argument('--current_batch', type=int, default=1, help="Current batch number.")
    parser.add_argument('--uv_material', nargs='+', default=[], help="List of UV materials.")
    parser.add_argument('--camera', nargs='+', default=[], help="List of camera names.")
    parser.add_argument('--dir_location', type=str, default='', help="Directory location.")

    args = parser.parse_args()
    input_directory = f"${args.dir_location}/all_mesh"

    num_processes = args.thread_num
    processes = []

    mesh_path_list = next(os.walk(input_directory))[1]

    mesh_list_batch = split_list_into_parts(mesh_path_list, args.batch_size, args.current_batch)

    dirs_per_process = len(mesh_list_batch) // num_processes
    remaining_dirs = len(mesh_list_batch) % num_processes

    start_idx = 0
    for i in range(num_processes):
        thread_blender_file = args.blend_file.replace('.blend', f'_{args.current_batch}_{i}.blend')
        output_blender_file_path = f"${args.dir_location}/blender/{thread_blender_file}"
        if not os.path.isfile(output_blender_file_path):
            shutil.copy(f"${args.dir_location}/blender/{args.blend_file}", output_blender_file_path)
        end_idx = start_idx + dirs_per_process + (1 if i < remaining_dirs else 0)
        process = BlenderProcess(i, mesh_list_batch[start_idx:end_idx], thread_blender_file, args.uv_material,
                                 args.camera, args.dir_location)
        processes.append(process)
        print(f"Start process: {thread_blender_file}")
        process.start()
        start_idx = end_idx

    for process in processes:
        process.join()


if __name__ == '__main__':
    tic = time.time()
    mp.set_start_method('spawn')
    main()
    toc = time.time()
    print(f'Copying meshes done, took {toc - tic} seconds.')
