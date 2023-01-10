import os

def read_and_sort_folders(directory):
    # Get a list of all subdirectories in the specified directory
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # Sort the subdirectories lexicographically
    subdirectories.sort()

    # Iterate through each subdirectory
    for subdir in subdirectories:
        # Construct the full path to the eval_results.txt file
        eval_file = os.path.join(directory, subdir, 'eval_results.txt')

        # Check if the file exists
        if os.path.isfile(eval_file):
            # Print the name of the subdirectory
            print(f'Folder: {subdir}')
            
            # Open and read the contents of the eval_results.txt file
            with open(eval_file, 'r') as f:
                eval_results = f.read()
                print(eval_results)
        else:
            # If the file does not exist, print an error message
            print(f'Error: {eval_file} does not exist.')
