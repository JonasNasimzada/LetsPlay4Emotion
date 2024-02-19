import csv


def process_csv(input_file, substrings):
    output1 = []
    output2 = []

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Assuming the first row contains headers
        output1.append(headers)
        output2.append(headers)

        for row in reader:
            found_substring = False
            for substring in substrings:
                if any(substring in cell for cell in row):
                    output1.append(row)
                    found_substring = True
                    break
            if not found_substring:
                output2.append(row)

    return output1, output2


def write_output(output_file, data):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def main():
    input_file = '../../model/datasets/old/train-multi.csv'
    substrings = ['100914_m_39', '101114_w_37', '082315_w_60', '083114_w_55', '083109_m_60', '072514_m_27',
                  '080309_m_29', '112016_m_25', '112310_m_20', '092813_w_24', '112809_w_23', '112909_w_20',
                  '071313_m_41', '101309_m_48', '101609_m_36', '091809_w_43', '102214_w_36', '102316_w_50',
                  '112009_w_43', '101814_m_58', '101908_m_61', '102309_m_61', '112209_m_51', '112610_w_60',
                  '112914_w_51', '120514_w_56']
    input_file1 = '../../model/datasets/old/val-binary.csv'
    input_file2 = '../../model/datasets/old/val-binary.csv'
    output_file = '../../model/datasets/train-binary.csv'
    output1, output2 = process_csv(input_file, substrings)

    # output_file1 = '../../model/datasets/val-multi.csv'
    output_file2 = '../../model/datasets/train-multi.csv'

    # write_output(output_file1, output1)
    write_output(output_file2, output2)

    print(f"Output files generated: {output_file2}")


if __name__ == "__main__":
    main()
