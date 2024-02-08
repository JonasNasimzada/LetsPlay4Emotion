import argparse
import glob
import os
from pathlib import Path

import torch.multiprocessing as mp
from tqdm import auto

import gdl
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from gdl_apps.EMOCA.utils.io import save_obj
from gdl_apps.EMOCA.utils.io import test
from gdl_apps.EMOCA.utils.load import load_model


def process_video_chunk(chunk_data):
    video_path, output_folder, emoca = chunk_data

    dm = TestFaceVideoDM(video_path, output_folder, processed_subfolder=None,
                         batch_size=4, num_workers=4)
    dm.prepare_data()
    dm.setup()

    dl = dm.test_dataloader()
    for j, batch in enumerate(auto.tqdm(dl)):
        current_bs = batch["image"].shape[0]
        img = batch
        vals, visdict = test(emoca, img)
        for i in range(current_bs):
            # name = f"{(j*batch_size + i):05d}"
            name = batch["image_name"][i]
            sample_output_folder = Path(output_folder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)
            print("here")
            save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)


def worker(queue):
    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break
        process_video_chunk(chunk_data)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, default='videos_directory',
                        help="Input directory containing video files.")
    parser.add_argument('--output_folder', type=str, default="image_output",
                        help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str,
                        default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default='detail', help="coarse or detail")
    parser.add_argument('--thread_num', type=int, default=1, help="number of threads")

    args = parser.parse_args()

    emoca, conf = load_model(args.path_to_models, args.model_name, args.mode)
    emoca.cuda()
    emoca.eval()

    num_processes = args.thread_num

    video_paths = glob.glob(os.path.join(args.input_directory, '*.mp4'))

    queue = mp.Queue()
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue,))
        processes.append(p)
        p.start()

    chunk_id = 0
    for video_path in video_paths:
        chunk_data = (video_path, args.output_folder, emoca)
        queue.put(chunk_data)
        chunk_id += 1

    for _ in range(num_processes):
        queue.put(None)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
