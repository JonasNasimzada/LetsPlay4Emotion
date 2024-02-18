import argparse
import os
import shutil
import subprocess
import time

import torch.multiprocessing as mp


def split_list_into_parts(input_list, n, y):
    # Calculate the length of each part
    part_length = len(input_list) // n

    # Initialize an empty list to store the parts
    parts = []

    # Iterate over the range of n
    for i in range(n):
        if i > y:
            continue
        # Calculate the start and end indices for the current part
        start_index = i * part_length
        end_index = (i + 1) * part_length if i < n - 1 else None

        # Slice the input list to extract the current part
        current_part = input_list[start_index:end_index]

        # Append the current part to the list of parts
        parts.append(current_part)

    # Return the y-th sublist
    return parts[y] if 0 <= y < n else None


class BlenderProcess(mp.Process):
    def __init__(self, process_number, directories, blender_file):
        super().__init__()
        self.blender_file = blender_file
        self.process_number = process_number
        self.directories = directories

    def run(self):
        print(f"Process {self.process_number} is running")
        self.execute_command()

    def execute_command(self):
        command = f"apptainer exec " \
                  f"--bind ~/blender/Stop-motion-OBJ:/usr/local/blender/3.6/scripts/addons/Stop-motion-OBJ " \
                  f"--bind ~/blender/config:/usr/local/blender/3.6/config/ " \
                  f"--bind /groups/constantin_students/jnasimzada/all_mesh:/usr/local/videos_input " \
                  f"--bind /groups/constantin_students/jnasimzada/videos_mesh:/usr/local/videos_output " \
                  f"--bind /groups/constantin_students/jnasimzada/LetsPlay4Emotion/utils/Blender:/usr/local/Blender_script" \
                  f"--bind /groups/constantin_students/jnasimzada/blender:/usr/local/blend_file" \
                  f"/Blender_script " \
                  f"--nv docker://blendergrid/blender:3.6.8 " \
                  f"/usr/local/blender/blender " \
                  f"--background " \
                  f"/usr/local/blend_file/{self.blender_file} " \
                  f"--python /usr/local/Blender_script/generate_mesh_video.py " \
                  f"--" \
                  f" --batch_files {' '.join(self.directories)}"
        try:
            subprocess.run(command, capture_output=True, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command in Process {self.process_number} for directory {self.directories}: {e}")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend_file', type=str)
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--current_batch', type=int, default=1)

    args = parser.parse_args()

    input_directory = "/groups/constantin_students/jnasimzada/all_mesh"

    num_processes = args.thread_num
    processes = []

    mesh_path_list = next(os.walk(input_directory))[1]

    mesh_list_batch = split_list_into_parts(mesh_path_list, args.batch_size, args.current_batch)

    dirs_per_process = len(mesh_list_batch) // num_processes
    remaining_dirs = len(mesh_list_batch) % num_processes

    start_idx = 0
    for i in range(num_processes):
        thread_blender_file = args.blend_file.replace('.blend', f'_{i}.blend')
        output_blender_file_path = f"/groups/constantin_students/jnasimzada/blender/{thread_blender_file}"
        if not os.path.isfile(output_blender_file_path):
            shutil.copy(f"/groups/constantin_students/jnasimzada/blender/{args.blend_file}",
                        output_blender_file_path)
        end_idx = start_idx + dirs_per_process + (1 if i < remaining_dirs else 0)
        process = BlenderProcess(i, mesh_list_batch[start_idx:end_idx], thread_blender_file)
        processes.append(process)
        process.start()
        start_idx = end_idx

    for process in processes:
        process.join()


if __name__ == '__main__':
    tic = time.time()
    mp.set_start_method('spawn')
    main()
    toc = time.time()
    print(f'coping meshes done, took {toc - tic} seconds.')
