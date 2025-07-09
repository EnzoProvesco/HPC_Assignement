"""Total (MB),Count,Avg (MB),Med (MB),Min (MB),Max (MB),StdDev (MB),Operation
"482,333",3,"160,778","160,778","160,778","160,778","0,000",[CUDA memcpy DtoH]
"321,555",2,"160,778","160,778","160,778","160,778","0,000",[CUDA memcpy HtoD]
From a text like thhis, which is obtained from the nsys profile of cuda, retrieve the info regarding memory and create a csv file with the following column:
FileName;Count;Avg;Min;Max;Operation 
"""
import csv
import sys
import os

def parse_nsys_output(input_text, path):
    lines = input_text.strip().split('\n')
    header = lines[0].split(',')
    data = []
    file_name = os.path.basename(path)
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) < 7:
            continue  # Skip lines that do not have enough data
        operation = parts[-1].strip('[]')
        count = parts[1].strip('"')
        avg = parts[2].strip('"')
        min_val = parts[4].strip('"')
        max_val = parts[5].strip('"')
        #
        data.append([file_name, parts[0], count, avg, min_val, max_val, operation])

    return data

def write_to_csv(data, writer):
        writer.writerow(['FileName', 'Count', 'Avg', 'Min', 'Max', 'Operation'])
        for row in data:
            writer.writerow(row)

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else ''
    output_file = sys.argv[2] if len(sys.argv) > 2 else "logMemory.csv"

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['FileName', 'Total (MB)', 'Count', 'Avg (MB)', 'Min (MB)', 'Max (MB)', 'Operation'])
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv'):
                input_file_path = os.path.join(root, file)
                with open(input_file_path, 'r') as f:
                    input_text = f.read()
                data = parse_nsys_output(input_text, input_file_path)
                write_to_csv(data, output_file)
    with open(input_file_path, 'r') as f:
        input_text = f.read()
    data = parse_nsys_output(input_text, input_file_path)
    write_to_csv(data, output_file)

if __name__ == "__main__":
    main()
