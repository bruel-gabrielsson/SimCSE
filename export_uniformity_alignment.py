import json

json_file_name = input("Enter the name of the JSON file: ")
output_file_name = input("Enter the name of the output text file: ")

# Open the JSON file and load its contents into a Python dictionary
with open(json_file_name) as json_file:
    data = json.load(json_file)

# Extract the 'log_history' array from the dictionary
log_history = data['log_history']

# Open the output file for writing
with open(output_file_name, 'w') as output_file:
    # Loop through the objects in the 'log_history' array
    for log_object in log_history:
        # Extract the 'alignment' and 'uniformity' values from the object
        #print(log_object)
        if "alignment" in log_object:
            alignment = log_object['alignment']
            uniformity = log_object['uniformity']
            # Write the alignment and uniformity values to the output file
            output_file.write(f"{alignment} {uniformity}\n")
