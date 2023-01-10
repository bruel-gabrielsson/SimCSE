import os
import sys
from collections import defaultdict

def read_and_sort_folders(directory):
    # Create a dictionary to store the subdirectory name and eval_stsb_spearman value
    eval_stsb_spearman = defaultdict(float)

    # Get a list of all subdirectories in the specified directory
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # Iterate through each subdirectory
    for subdir in subdirectories:
        # Construct the full path to the eval_results.txt file
        eval_file = os.path.join(directory, subdir, 'eval_results.txt')

        # Check if the file exists
        if os.path.isfile(eval_file):
            with open(eval_file, 'r') as f:
                # Read the file line by line
                for line in f:
                    # Check if the line contains the eval_stsb_spearman attribute
                    if "eval_avg_sts" in line:
                        # Extract the value of eval_stsb_spearman
                        eval_stsb_spearman[subdir] = float(line.split()[-1])
                        break
        else:
            # If the file does not exist, print an error message
            print(f'Error: {eval_file} does not exist.')

    # Sort the dictionary based on the eval_stsb_spearman value in descending order
    sorted_folders = dict(sorted(eval_stsb_spearman.items(), key=lambda item: item[1], reverse=True))

    # Print the subdirectory name and eval_stsb_spearman value
    for subdir, val in sorted_folders.items():
        print(f'{subdir} : {val}')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if os.path.isdir(sys.argv[1]):
            read_and_sort_folders(sys.argv[1])
        else:
            print(f"{sys.argv[1]} is not a directory.")
    else:
        print("Please provide a directory path as an argument")
