import argparse
import os
import shutil
import subprocess
import time

import torch.multiprocessing as mp


class CommandProcess(mp.Process):
    def __init__(self, process_number, directories, blender_file):
        super().__init__()
        self.blender_file = blender_file
        self.process_number = process_number
        self.directories = directories

    def run(self):
        print(f"Process {self.process_number} is running")
        for directory in self.directories:
            if os.path.isdir(directory):
                self.execute_command(directory)

    def execute_command(self, directory):
        command = f"apptainer exec " \
                  f"--bind ~/blender/Stop-motion-OBJ:/usr/local/blender/3.6/scripts/addons/Stop-motion-OBJ " \
                  f"--bind ~/blender/config:/usr/local/blender/3.6/config/ " \
                  f"--nv " \
                  f"docker://blendergrid/blender:3.6.8 /usr/local/blender/blender " \
                  f"--background {self.blender_file} " \
                  f"--python ./generate_mesh_video.py"
        try:
            subprocess.run(['ls', '-l', directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
            print(f"Process {self.process_number} executed command for directory {directory}: ls -l {directory}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command in Process {self.process_number} for directory {directory}: {e}")


def worker(queue):
    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--blend_file', type=str)
    parser.add_argument('--thread_num', type=int, default=1, help="number of threads")

    args = parser.parse_args()

    num_processes = args.thread_num
    processes = []

    mesh_path_list = next(os.walk(args.input_directory))[1]
    dirs_per_process = len(mesh_path_list) // num_processes
    remaining_dirs = len(mesh_path_list) % num_processes

    start_idx = 0
    for i in range(num_processes):
        thread_blender_file = args.blend_file.replace('.blend', f'_{i}.blend')
        shutil.copy(args.blend_file, thread_blender_file)
        end_idx = start_idx + dirs_per_process + (1 if i < remaining_dirs else 0)
        process = CommandProcess(i, mesh_path_list[start_idx:end_idx], thread_blender_file)
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
