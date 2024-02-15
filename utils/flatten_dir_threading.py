import argparse
import os
import shutil
import time

import torch.multiprocessing as mp


def flatten_dir(chunk_data):
    input_dir, mesh_dir, destination_dir = chunk_data
    source_path = f"{input_dir}/{mesh_dir}"
    destination_path = f"{destination_dir}/{mesh_dir}"
    os.makedirs(destination_path, exist_ok=True)
    for root, dirs, files in os.walk(source_path):
        for file in files:
            # Check if the file ends with ".obj"
            if file.endswith("mesh_coarse.obj"):
                new_filename = os.path.join(os.path.basename(os.path.dirname(file)), file)
                new_filename = new_filename.replace(os.path.sep, "_")
                new_file_path = f"{destination_path}/{new_filename}"

                # Full path of the input file
                input_file_path = os.path.join(root, file)

                # Full path of the output file
                output_file_path = os.path.join(destination_path, new_file_path)

                # Copy the file to the output directory with the new filename
                shutil.copy(input_file_path, output_file_path)


def worker(queue):
    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break
        flatten_dir(chunk_data)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--thread_num', type=int, default=1, help="number of threads")

    args = parser.parse_args()

    num_processes = args.thread_num

    mesh_path_list = next(os.walk(args.input_directory))[1]

    queue = mp.Queue()
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue,))
        processes.append(p)
        p.start()

    chunk_id = 0
    for mesh_path in mesh_path_list:
        chunk_data = (args.input_directory, mesh_path, args.output_dir)
        queue.put(chunk_data)
        chunk_id += 1

    for _ in range(num_processes):
        queue.put(None)

    for p in processes:
        p.join()


if __name__ == '__main__':
    tic = time.time()
    mp.set_start_method('spawn')
    main()
    toc = time.time()
    print(f'coping meshes done, took {toc - tic} seconds.')
