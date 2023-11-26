import os
import re
import numpy as np

def extract_number(dir_name):
    match = re.search(r'SUPREG_(\d+)_', dir_name)
    return int(match.group(1)) if match else None

def get_eval_stsb_spearman(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if "eval_stsb_spearman" in line:
                    return float(line.split()[-1])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return None

def iterate_directories(base_dir):
    data = {}

    for root, dirs, _ in os.walk(base_dir):
        print("root", root)
        if 'SUPER_REG_NOV23_S' in root:
            print("root", root)
            for dir in dirs:
                num = extract_number(dir)
                if num is not None:
                    file_path = os.path.join(root, dir, 'train_results.txt')
                    spearman_value = get_eval_stsb_spearman(file_path)
                    if spearman_value is not None:
                        if num not in data:
                            data[num] = []
                        data[num].append(spearman_value)
                else:
                    print(f"Could not extract number from {dir}")

    return data

def compute_statistics(data):
    stats = {}
    for num, values in data.items():
        stats[num] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return stats

# Example usage
#base_dir = "/mnt/brg/simcse-data/HYPER/"
base_dir = "/skunk-pod-storage-brg-40mit-2eedu-pvc/DATA/simcse-data/HYPER/"
data = iterate_directories(base_dir)
stats = compute_statistics(data)
for num, value in stats.items():
    print(f"Number {num}: Mean = {value['mean']}, Std = {value['std']}")
