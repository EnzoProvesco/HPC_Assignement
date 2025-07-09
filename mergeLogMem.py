import csv
import sys
import os

def strip_commas(input_string):
    #check if a comma is inside a "" which a lot of character in between the ""
    in_quotes = False
    result = []
    for char in input_string:
        if char == '"':
            in_quotes = not in_quotes
        if char == ',' and in_quotes:
            result.append('.')
        else:
            result.append(char)
    return ''.join(result)

def parse_nsys_output(input_text, filepath):
    lines = input_text.strip().split('\n')
    data = []
    #replace all the comma insiede a "" into dot
    
    for line in lines[1:]:  # Skip the header line
        line = strip_commas(line)
        print(f"Processing line: {line}")
        parts = line.split(',')
        if len(parts) < 7:
            continue  # Skip lines that do not have enough data

        # Extract relevant fields
        total_mb = parts[0].strip('"')
        count = parts[1].strip('"')
        avg_mb = parts[2].strip('"')
        min_mb = parts[4].strip('"')
        max_mb = parts[5].strip('"')
        operation = parts[7].strip('"')
        filename = os.path.basename(filepath)
        # Append the parsed data to the list
        data.append([filename, total_mb, count, avg_mb, min_mb, max_mb, operation])
        print(f"Parsed data: {data[-1]}")
    return data

def main():
    """
    Main function to find, process, and merge all CSV reports.
    """
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_directory> <output_csv_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isdir(input_path):
        print(f"Error: Input directory not found at '{input_path}'")
        sys.exit(1)

    # Open the output file in write mode to start fresh.
    # The 'with' block now encloses the entire loop.
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        
        # Write the header row only ONCE
        writer.writerow(['FileName', 'Total (MB)', 'Count', 'Avg (MB)', 'Min (MB)', 'Max (MB)', 'Operation'])

        # Use os.walk to recursively find all files in the input directory
        for root, _, files in os.walk(input_path):
            for filename in files:
                # Process only files that end with .csv or .txt
                if filename.endswith('.csv') or filename.endswith('.txt'):
                    input_file_path = os.path.join(root, filename)
                    print(f"Processing file: {input_file_path}")
                    
                    try:
                        with open(input_file_path, 'r') as f_in:
                            input_text = f_in.read()
                            print(input_text)
                        # Parse the content of the current file
                        data = parse_nsys_output(input_text, input_file_path)
                        print(data)
                        # Write the parsed data rows directly to the main output file
                        if data:
                            writer.writerows(data)
                    except Exception as e:
                        print(f"Could not process file {input_file_path}: {e}")

    print(f"\nProcessing complete. Aggregated data saved to {output_file}")

if __name__ == "__main__":
    main()