import pandas as pd
import matplotlib.pyplot as plt


def count_names_in_column(csv_file, name):
    try:
        # Read the CSV file into a pandas DataFrame with tab delimiter and without header
        df = pd.read_csv(csv_file, delimiter='\t', header=None)

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

        # Plot a bar chart
        name_counts.plot(kind='bar')
        plt.xlabel('label')
        plt.ylabel('Count')
        plt.title('Counts of Names')
        plt.xticks(rotation=0, ha='right')
        plt.tight_layout()
        plt.savefig(fname=name)
        plt.show()

    except FileNotFoundError:
        print("File not found. Please provide a valid CSV file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Prompt the user for the CSV file path and column name
    csv_file_path = "/Users/joni/LetsPlay4Emotion/model/datasets/val-binary.csv"
    csv_name = csv_file_path.split('/')[-1].split('.')[0]

    # Call the function to count names in the specified column
    count_names_in_column(csv_file_path, csv_name)
