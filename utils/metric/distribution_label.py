import os

import pandas as pd
import matplotlib.pyplot as plt


def count_names_in_column(csv_file, name):
    try:
        # Read the CSV file into a pandas DataFrame with tab delimiter and without header
        df = pd.read_csv(csv_file, delim_whitespace=True, header=None)

        # Assign default column names if the DataFrame doesn't have column names
        if df.columns[0] == 0:
            df.columns = [f'Column{i + 1}' for i in range(len(df.columns))]

        # Check if the DataFrame has at least two columns
        if len(df.columns) < 2:
            print("CSV file should have at least two columns for name counting.")
            return

        # Select the second column for name counting
        column_name = df.columns[1]

        # Count the occurrences of each unique value in the second column
        name_counts = df[column_name].value_counts()

        # Print the counts
        print("Counts of names in the second column:")
        print(name_counts)
        plt.rcParams.update({'font.size': 19})
        # Plot a bar chart
        name_counts.plot(kind='bar')
        plt.ylabel('Amount of videos')
        plt.xlabel('Pain categories')
        plt.xticks(rotation=0, ha='right')
        plt.tight_layout()
        plt.savefig(fname=name)
        plt.clf()

    except FileNotFoundError:
        print("File not found. Please provide a valid CSV file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    input_file_dir = os.listdir("input_csv/")
    for file in input_file_dir:
        csv_name = file.split('/')[-1].split('.')[0]
        count_names_in_column(f"./test_csv/{file}", f"./output/{csv_name}")
