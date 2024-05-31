import os

# Define the paths
labels_path = r'E:\\INTERNSHIP\\SYOOK\\ppe_data\\val\\labels'
output_labels_path = r'E:\\INTERNSHIP\\SYOOK\\ppe_data\\val\\filter_labels'

# Create output directory if it doesn't exist
os.makedirs(output_labels_path, exist_ok=True)

# Class to keep (e.g., class 0 for 'person')
target_class = [1,2,3,4,5,6,7,8,9]

def filter_labels(labels_path, output_labels_path, target_class):
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            input_file = os.path.join(labels_path, label_file)
            output_file = os.path.join(output_labels_path, label_file)
            
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                for line in infile:
                    class_id = int(line.split()[0])
                    if class_id in target_class:
                        outfile.write(line)

# Filter the labels
filter_labels(labels_path, output_labels_path, target_class)
