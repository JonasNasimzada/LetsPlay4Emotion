import argparse
import os
import shutil
import time

import torch.multiprocessing as mp


def process_video_chunk(chunk_data):
    source_dir, destination_dir = chunk_data
    for item in os.listdir(source_dir):
        # Construct the full path of the item
        item_path = os.path.join(source_dir, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Copy the directory to the destination directory
            shutil.copytree(item_path, os.path.join(destination_dir, item))


def worker(queue):
    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break
        process_video_chunk(chunk_data)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--thread_num', type=int, default=1, help="number of threads")

    args = parser.parse_args()

    num_processes = args.thread_num

    mesh_path_list = os.listdir(args.input_directory)

    queue = mp.Queue()
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue,))
        processes.append(p)
        p.start()

    chunk_id = 0
    for mesh_path in mesh_path_list:
        chunk_data = (mesh_path, args.output_dir)
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
