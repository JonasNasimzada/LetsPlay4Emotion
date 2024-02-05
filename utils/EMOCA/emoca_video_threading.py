import argparse
import glob
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import auto

import gdl
from gdl.datasets.ImageTestDataset import TestData
from gdl_apps.EMOCA.utils.io import save_obj
from gdl_apps.EMOCA.utils.io import test
# ... (other functions, classes, etc.)
from gdl_apps.EMOCA.utils.load import load_model


def process_video_chunk(chunk_data):
    """Processes a specified chunk of a video using a single GPU."""
    video_path, start_frame, end_frame, output_folder, model_name, path_to_models, mode, save_images, save_codes, save_mesh, process_id, gpu_id = chunk_data

    with torch.cuda.device(gpu_id):  # Assign chunk to specific GPU
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()

        dataset = TestData(video_path, face_detector="fan", max_detection=20, start_frame=start_frame,
                           end_frame=end_frame)

        # ... (process the chunk using the model and dataset on the assigned GPU)
        for j in auto.tqdm(range(len(dataset)),
                           desc=f"Processing chunk ({process_id}:{gpu_id}) {os.path.basename(video_path)} ({start_frame}-{end_frame})"):
            batch = dataset[j]
            vals, visdict = test(emoca, batch)

            current_bs = batch["image"].shape[0]

            for i in range(current_bs):
                name = batch["image_name"][i]
                sample_output_folder = Path(output_folder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)
                save_obj(emoca, str(sample_output_folder / f"{video_path.replace('/', '_')}_mesh.obj"), vals, i)


def worker(queue):
    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break
        process_video_chunk(chunk_data)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    # ... (add your argument parsing logic here)
    parser.add_argument('--input_directory', type=str, default='videos_directory',
                        help="Input directory containing video files.")
    parser.add_argument('--output_folder', type=str, default="image_output",
                        help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str,
                        default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--save_images', type=bool, default=False, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False,
                        help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=True, help="If true, output meshes will be saved")
    parser.add_argument('--mode', type=str, default='detail', help="coarse or detail")

    args = parser.parse_args()

    num_processes = min(mp.cpu_count(), len(torch.cuda.devices()))
    num_gpus_per_process = 1

    video_paths = glob.glob(os.path.join(args.input_directory, '*.mp4'))

    chunk_size = 64  # Adjust based on video length and resource constraints
    total_chunks = 0
    for video_path in video_paths:
        video_length_frames = len(TestData(video_path)) * 25  # Assuming 25fps
        total_chunks += (video_length_frames + chunk_size - 1) // chunk_size

    queue = mp.Queue()
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue,))
        processes.append(p)
        p.start()

    chunk_id = 0
    for video_path in video_paths:
        video_length_frames = len(TestData(video_path)) * 25
        for i in range((video_length_frames + chunk_size - 1) // chunk_size):
            start_frame = i * chunk_size
            end_frame = min((i + 1) * chunk_size - 1, video_length_frames - 1)
            process_id = chunk_id % num_processes
            gpu_id = chunk_id // (num_processes * num_gpus_per_process) % len(torch.cuda.devices())
            chunk_data = (
                video_path, start_frame, end_frame, args.output_folder, args.model_name, args.path_to_models, args.mode,
                args.save_images, args.save_codes, args.save_mesh, process_id, gpu_id)
            queue.put(chunk_data)
            chunk_id += 1

    for _ in range(num_processes):
        queue.put(None)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
