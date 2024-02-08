import threading

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData
import gdl
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
from concurrent.futures import ThreadPoolExecutor
import glob

# Define a lock to ensure thread safety
video_processing_lock = threading.Lock()


def process_video(video_path, output_folder, model_name, path_to_models, mode, save_images, save_codes, save_mesh):
    with video_processing_lock:
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()

        dataset = TestData(video_path, face_detector="fan", max_detection=20)

        for i in auto.tqdm(range(len(dataset))):
            batch = dataset[i]
            vals, visdict = test(emoca, batch)

            current_bs = batch["image"].shape[0]

            for j in range(current_bs):
                name = batch["image_name"][j]
                sample_output_folder = Path(output_folder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)

                if save_mesh:
                    save_obj(emoca, str(sample_output_folder / f"{video_path.replace('/', '_')}_mesh.obj"), vals, j)
                if save_images:
                    save_images(output_folder, name, visdict, with_detection=True, i=j)
                if save_codes:
                    save_codes(Path(output_folder), name, vals, i=j)

        print(f"Processing of {video_path} is complete")


def process_videos_in_thread(video_paths, output_folder, model_name, path_to_models, mode, save_images, save_codes,
                             save_mesh):
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_video, video, output_folder, model_name, path_to_models, mode, save_images,
                                   save_codes, save_mesh) for video in video_paths]

        # Wait for all threads to finish
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, default='videos_directory',
                        help="Input directory containing video files.")
    parser.add_argument('--output_folder', type=str, default="image_output",
                        help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str,
                        default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False,
                        help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    parser.add_argument('--mode', type=str, default='detail', help="coarse or detail")

    args = parser.parse_args()

    input_directory = args.input_directory
    output_folder = args.output_folder
    model_name = args.model_name
    path_to_models = args.path_to_models
    mode = args.mode

    videos = glob.glob(os.path.join(input_directory, '*.mp4'))

    process_videos_in_thread(videos, output_folder, model_name, path_to_models, mode, args.save_images, args.save_codes,
                             args.save_mesh)


if __name__ == '__main__':
    main()
