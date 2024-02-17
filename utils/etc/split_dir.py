import argparse
import os
import shutil


def split_directory(source_dir, n):
    # Create n new directories
    for i in range(n):
        new_dir = os.path.join(source_dir, f"subdir_{i}")
        os.makedirs(new_dir, exist_ok=True)

    # Get list of files in the source directory
    folders = next(os.walk(source_dir))[1]

    # Calculate how many files each subdirectory should contain
    folders_per_dir = len(folders) // n
    remainder = len(folders) % n

    # Distribute files among the subdirectories
    for i in range(n):
        start_index = i * folders_per_dir
        end_index = (i + 1) * folders_per_dir if i < n - 1 else None
        if i < remainder:
            end_index += 1
        folders_in_subdir = folders[start_index:end_index]
        for folder_name in folders_in_subdir:
            src_folder = os.path.join(source_dir, folder_name)
            dest_dir = os.path.join(source_dir, f"subdir_{i}")
            shutil.copytree(src_folder, os.path.join(dest_dir, folder_name))


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str)
    parser.add_argument('--dirs', type=int)

    args = parser.parse_args()

    split_directory(args.input_directory, args.dirs)
