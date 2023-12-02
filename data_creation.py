import os
import re
import subprocess
import sys

import cv2
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip


def perform_operation(file_path, csv_path):
    print(f"Performing operations on file: {file_path} and temperature-csv: {csv_path}")


def create_folders(output_folder, file_name):
    folders = ["png", "obj", "landmarks", "mesh"]
    for folder_type in folders:
        folder_path = os.path.join(output_folder, folder_type, file_name)
        os.makedirs(folder_path, exist_ok=True)


def extract_frames_from_csv(mp4_file, csv_file, output_folder, command):
    print("Creating .png files from the video clip and the temperature csv")
    df = pd.read_csv(csv_file, names=["time", "temperature"], index_col="time", delimiter='\t', header=None, skiprows=1)
    file_name = os.path.basename(os.path.splitext(mp4_file)[0])
    create_folders(output_folder, file_name)

    video_clip = VideoFileClip(mp4_file)

    for index, row in df.iterrows():
        start_time = int(index) / 1000000
        frame = video_clip.get_frame(start_time)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_name = f"frame_{file_name}_{index}.jpeg"
        output_path_png = os.path.join(output_folder, "png", file_name, frame_name)
        print(f"Creating frame {frame_name}")
        cv2.imwrite(output_path_png, frame_rgb)
        output_folder_landmarks = os.path.join(output_folder, "landmarks", file_name)
        output_folder_mesh = os.path.join(output_folder, "mesh", file_name)
        output_folder_obj = os.path.join(output_folder, "obj", file_name)
        print(f"Creating landmarks for file {frame_name}")
        subprocess.run(["python", command, output_path_png, output_folder_landmarks, "445_landmarks"])

        print(f"Creating mesh for file {frame_name}")
        subprocess.run(["python", command, output_path_png, output_folder_mesh, "head_mesh"])

        print(f"Creating .obj file for file {frame_name}")
        subprocess.run(["python", command, output_path_png, output_folder_obj, "3d_mesh"])
        print()

    video_clip.reader.close()


def process_folders(regex_person, regex_mp4, mp4_file_location, temperature_file_location, output_dir_path, command):
    for root, dirs, files in os.walk(mp4_file_location):
        for folder in dirs:
            if re.search(regex_person, folder):
                folder_path = os.path.join(root, folder)
                for mp4_file in os.listdir(folder_path):
                    if re.search(regex_mp4, mp4_file):
                        file_name, _ = os.path.splitext(mp4_file)
                        temperature_file_path = os.path.join(temperature_file_location, folder, f"{file_name}_temp.csv")
                        if not os.path.exists(temperature_file_path):
                            print(f"[WARNING]: {mp4_file} doesn't have a temperature csv file")
                            continue
                        mp4_file_path = os.path.join(mp4_file_location, folder, mp4_file)
                        perform_operation(mp4_file_path, temperature_file_path)
                        extract_frames_from_csv(mp4_file_path, temperature_file_path, output_dir_path, command)


if __name__ == "__main__":
    # file_regex = ".*PA4.*"
    file_regex = ".*"
    folder_regex = sys.argv[1]
    mp4_files = "/homes/jnasimzada/schmerzvideos/video"
    temperature_files = "/homes/jnasimzada/schmerzvideos/temperature"
    output_dir = "/homes/jnasimzada/output_data_3"
    python_command = "/homes/jnasimzada/DAD-3DHeads/demo.py"

    process_folders(regex_person=folder_regex, regex_mp4=file_regex, mp4_file_location=mp4_files,
                    temperature_file_location=temperature_files, output_dir_path=output_dir, command=python_command)
