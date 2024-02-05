import re

import cv2
import sys
import shutil

import os
import argparse

sys.path.append("./stargan-v2/core")
sys.path.append("./stargan-v2")
from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_test_loader

from core.solver import Solver
from utils.old import detect_and_crop_head


def create_frames(args, video_path):
    basename = video_path[:video_path.rfind('.')]
    output_folder, output_folder_copy = get_gender_paths(f"{args.output_frame_path}/{basename}")
    # Open the video file
    cap = cv2.VideoCapture(f"{args.video_path}/{video_path}")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_copy, exist_ok=True)

    # Get the frames and save each one
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of digits needed for leading zeros
    num_digits = len(str(total_frames))
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        # Save the frame
        frame_name = f"frame_{frame_count:0{num_digits}d}.png"
        frame_path = os.path.join(output_folder, frame_name)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_and_crop_head.detect_and_crop_head_and_interpolate(args, frame, frame_path)
        if frame_count == 0:
            shutil.copy(frame_path, os.path.join(output_folder_copy, frame_name))
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"{frame_count} frames saved to {output_folder}")
    frames = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    frames.sort()
    print("frames in folder:")
    print(frames)
    print()
    # shutil.copy(f"{output_dir}/{basename}/{basename}", f"{output_dir}/{basename}/{basename}-copy")
    return basename, total_frames


def create_video(input_dir, output_video_path, frame_rate=25.0):
    # Get the list of frame files in the input directory
    frames = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    if not frames:
        print("No frames found in the input directory.")
        return

    # Sort frames based on their names
    frames.sort()
    print("create video for this frames:")
    print(frames)

    # Read the first frame to get frame dimensions
    first_frame_path = os.path.join(input_dir, frames[0])
    first_frame = cv2.imread(first_frame_path)
    original_height, original_width, _ = first_frame.shape

    # Set the target dimensions
    target_width, target_height, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (target_width, target_height))

    # Resize and write each frame to the video
    for frame_name in frames:
        frame_path = os.path.join(input_dir, frame_name)
        frame = cv2.imread(frame_path)

        # Resize the frame to the target dimensions
        resized_frame = cv2.resize(frame, (target_width, target_height))

        video_writer.write(resized_frame)

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video created at: {output_video_path}")


def print_frame_rate(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate
    frame = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    print(f"Frame rate of the video {video_path}: {frame} frames per second")
    return frame


def get_gender_paths(dir_name):
    gender = 'male' if re.search(r'_m_', dir_name) else 'female'
    return f"{dir_name}/{gender}", f"{dir_name}/{'female' if gender == 'male' else 'male'}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # arguments from stargan-v2
    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')
    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')
    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='stargan-v2/expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='stargan-v2/expr/checkpoints',
                        help='Directory for saving network checkpoints')
    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='stargan-v2/expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')
    # directory for testing
    parser.add_argument('--result_dir', type=str, default='stargan-v2/expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')
    # face alignment
    parser.add_argument('--wing_path', type=str, default='stargan-v2/expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='stargan-v2/expr/checkpoints/celeba_lm_mean.npz')
    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)
    #####################
    # my arguments
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_frame_path_video', type=str, required=True)
    parser.add_argument('--output_frame_path', type=str, required=True, default="./output_test")
    parser.add_argument('--output_gen_video', type=str, default="test_video.mp4")
    args = parser.parse_args()
    cudnn.benchmark = True
    torch.manual_seed(777)
    solver = Solver(args)

    video_dir = [f for f in os.listdir(args.video_path) if f.endswith(".mp4")]
    for video in video_dir:
        print(f"video file : {video}")
        # 1. frames will be created
        base_name, max_frames = create_frames(args, video)
        # 2. images will be interpolated and added to different person but with the same mimik
        output_dir = f"{args.output_frame_path}/{base_name}"
        batch_size = max_frames + 1

        loaders = Munch(src=get_test_loader(root=output_dir,
                                            img_size=args.img_size,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample_solo_picture_file(loaders, base_name, max_frames)
    # solver.sample(loaders, chunk_counter)

    # fps = print_frame_rate(args.video_path)
    # create_video(f"{args.result_dir}/{base_name}", "test_result_video.mp4")
    # create_video(args.out_dir, "test_interpol_video.mp4")
    for root, dirs, files in os.walk(args.result_dir):
        if not dirs:
            working_dir = os.path.relpath(root, args.result_dir)
            print(f"dir for video:{os.path.dirname(args.result_dir)}/{working_dir}")
            result_dir_video = f"{os.path.dirname(args.result_dir)}/{working_dir}/stargan_result_video.mp4"
            create_video(working_dir, result_dir_video)
