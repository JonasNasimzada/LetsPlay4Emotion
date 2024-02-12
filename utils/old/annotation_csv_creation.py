import argparse
import os
import re

import numpy as np
import pandas as pd

import torch.nn as nn
from pytorch_lightning import Trainer
from torch.nn import BCEWithLogitsLoss


def perform_operation(file_path, csv_path, output_path):
    print(
        f"Performing operations on file: {file_path} and temperature-csv: {csv_path} and will be saved at {output_path}")


def process_folders(regex_person, regex_file, bio_file_location, temperature_file_location, output_dir_path):
    for root, dirs, files in os.walk(bio_file_location):
        for folder in sorted(dirs):
            if re.search(regex_person, folder):
                folder_path = os.path.join(root, folder)
                for bio_file in os.listdir(folder_path):
                    if re.search(regex_file, bio_file):
                        pattern = re.compile(r'^(.*?)(?=_bio)')
                        match = pattern.search(str(bio_file))
                        file_designation = match.group(0)
                        temperature_file_path = os.path.join(temperature_file_location, folder,
                                                             f"{file_designation}_temp.csv")
                        if not os.path.exists(temperature_file_path):
                            print(f"[WARNING]: {bio_file} doesn't have a temperature csv file")
                            continue
                        output_file_path = os.path.join(output_dir_path,
                                                        f"{file_designation}.csv")
                        perform_operation(bio_file_location, temperature_file_path, output_path=output_file_path)
                        bio_file_path = os.path.join(bio_file_location, folder, bio_file)
                        merge_csv(temperature_file=temperature_file_path, bio_file=bio_file_path,
                                  output_file=output_file_path)


def merge_csv(temperature_file, bio_file, output_file):
    # Read CSV files into Pandas DataFrames
    df1 = pd.read_csv(temperature_file, names=["time", "temperature"], delimiter='\t', header=None, skiprows=1)

    df2 = pd.read_csv(bio_file, names=["time", "gsr", "ecg", "emg_trapezius", "emg_corrugator", "emg_zygomaticus"],
                      delimiter='\t', header=None, skiprows=1)
    # Merge DataFrames on the key column
    merged_df = pd.merge(df1, df2, on='time', how='left')
    columns_to_format = ['gsr', 'ecg', 'emg_trapezius']

    # Convert the specified columns to scientific notation
    merged_df[columns_to_format] = merged_df[columns_to_format].applymap(lambda x: np.format_float_scientific(x))

    # Save the merged DataFrame to a new CSV file

    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bio_dir", required=True)
    parser.add_argument("--temp_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--regex_Person", default="", required=False)
    parser.add_argument("--regex_File", default="", required=False)

    args = parser.parse_args()

    # Replace these file paths with the actual paths of your CSV files
    # bio_dir = '/Users/joni/Downloads/biosignals_filtered'
    # temp_dir = '/Users/joni/Downloads/create_data/temperature'
    # output_dir = '/Users/joni/Downloads/create_data/output'

