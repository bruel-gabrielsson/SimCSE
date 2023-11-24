def process_file_with_floats(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            # Remove spaces and tabs, then check if the line contains only digits, commas, and periods
            cleaned_line = line.replace(" ", "").replace("\t", "")
            if all(char.isdigit() or char in {',', '.'} for char in cleaned_line.strip()):
                # Convert the line to a list of floats
                numbers = [float(num) for num in cleaned_line.strip().split(',')]
                # Calculate mean and standard deviation
                mean = sum(numbers) / len(numbers)
                std_dev = (sum((x - mean) ** 2 for x in numbers) / len(numbers)) ** 0.5
                # Print the results
                print(f"{mean}) +- (0,{std_dev})")
            else:
                # Print the original line
                print(line.strip())

# Example usage
# process_file_with_floats('path_to_file.txt')
# This function now processes lines with floats (including periods).




if __name__ == '__main__':
    process_file_with_floats('/skunk-pod-storage-brg-40mit-2eedu-pvc/SimCSE/export_mean_std.txt')