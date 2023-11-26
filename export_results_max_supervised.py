import os
import sys
from collections import defaultdict
import re

def read_and_sort_folders(directory):
    def regex(x):
        last = ""
        try:
            last = re.findall(r'/(.*?)', x)[-1]
        except:
            last = x
            print("err", x)
        try:
            #last = "_".join(re.findall(r'(.*?)_(.*?)_(.*?)_', last)[0])
            last = "_".join(re.findall(r'(.*?)_(.*?)_', last)[0])
        except:
            last = ""
            print("err", x)
        return last

    # Create a dictionary to store the subdirectory name and eval_stsb_spearman value
    eval_stsb_spearman = defaultdict(float)
    eval_avg_sts = defaultdict(float)
    eval_avg_transfer = defaultdict(float)
    eval_dir = defaultdict(str)


    # Get a list of all subdirectories in the specified directory
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # Iterate through each subdirectory
    for subdir in subdirectories:
        # Construct the full path to the eval_results.txt file
        eval_file = os.path.join(directory, subdir, 'train_results.txt')

        # Check if the file exists
        if os.path.isfile(eval_file):
            with open(eval_file, 'r') as f:
                # Read the file line by line
                this_eval_stsb_spearman, this_eval_avg_sts, this_eval_avg_transfer = 0, 0, 0
                for line in f:
                    # Check if the line contains the eval_stsb_spearman attribute
                    if "eval_stsb_spearman" in line:
                        this_eval_stsb_spearman = float(line.split()[-1]) # Extract the value of eval_stsb_spearman
                    if "eval_avg_sts" in line:
                        this_eval_avg_sts = float(line.split()[-1]) # Extract the value of eval_stsb_spearman
                    if "eval_avg_transfer" in line:
                        this_eval_avg_transfer= float(line.split()[-1]) # Extract the value of eval_stsb_spearman


                    
                this_prefix = directory + subdir # regex(subdir)
                if this_eval_stsb_spearman > eval_stsb_spearman[this_prefix]:
                    eval_stsb_spearman[this_prefix] = this_eval_stsb_spearman
                    #eval_avg_sts[this_prefix] = this_eval_avg_sts
                    #eval_avg_transfer[this_prefix] = this_eval_avg_transfer
                    eval_dir[this_prefix] = eval_file
        else:
            # If the file does not exist, print an error message
            print(f'Error: {eval_file} does not exist.')

    # Sort the dictionary based on the eval_stsb_spearman value in descending order
    sorted_folders = dict(sorted(eval_stsb_spearman.items(), key=lambda item: item[1], reverse=True))

    # Print the subdirectory name and eval_stsb_spearman value
    for prefix, val in sorted_folders.items():
        print("PREFIX: {}\n{}\n{}\n{}\n{}\n".format(prefix, eval_stsb_spearman[prefix], eval_avg_sts[prefix], eval_avg_transfer[prefix], eval_dir[prefix]))

if __name__ == '__main__':
    
    

    if len(sys.argv) == 2:
        if os.path.isdir(sys.argv[1]):
            read_and_sort_folders(sys.argv[1])
        else:
            print(f"{sys.argv[1]} is not a directory.")
    else:
        print("Please provide a directory path as an argument")
