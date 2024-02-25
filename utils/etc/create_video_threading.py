import argparse
import os
import time

import cv2
import torch.multiprocessing as mp


def create_video(chunk_data):
    frame_path, video_name, output_path = chunk_data
    fps = 25
    png_files = sorted([file for file in os.listdir(frame_path) if file.endswith('.png')])
    image = cv2.imread(f"{frame_path}/{png_files[0]}")
    height, width, _ = image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{output_path}/{video_name}", fourcc, fps, (height, width))
    for frame in png_files:
        frame = cv2.imread(f"{frame_path}/{frame}")
        video.write(frame)
    video.release()


def worker(queue):
    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break
        create_video(chunk_data)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--thread_num', type=int, default=1, help="number of threads")

    args = parser.parse_args()

    num_processes = args.thread_num

    queue = mp.Queue()
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue,))
        processes.append(p)
        p.start()

    chunk_id = 0

    os.makedirs(args.output_dir, exist_ok=True)

    for root, dirs, files in os.walk(args.input_directory):
        for directory in dirs:
            if directory.lower() == "detections":
                chunk_data = (f"{root}/{directory}", os.path.basename(root), args.output_dir)
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