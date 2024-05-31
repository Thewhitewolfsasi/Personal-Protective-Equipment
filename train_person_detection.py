# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator
# # Define dataset paths
# train_data = r'E:\\INTERNSHIP\\SYOOK\\person.yaml'
# # Initialize YOLOv8 model
# model = YOLO('yolov8n.pt')
# model.predict()
 
import os
import matplotlib.pyplot as plt
from collections import Counter

def count_classes(labels_path):
    class_counts = Counter()
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_path, label_file), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts

def plot_class_distribution(class_counts, title):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    print(classes, counts)
    
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, tick_label=classes)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title(title)
    plt.show()

# Define the paths
train_labels_path = r'E:\\INTERNSHIP\\SYOOK\\datasets\\ppe_data\\train\\labels'
val_labels_path = r'E:\\INTERNSHIP\\SYOOK\\datasets\\ppe_data\\val\\labels'

# Count the classes
train_class_counts = count_classes(train_labels_path)
val_class_counts = count_classes(val_labels_path)

# Plot the distributions
plot_class_distribution(train_class_counts, 'Class Distribution in Training Dataset')
plot_class_distribution(val_class_counts, 'Class Distribution in Validation Dataset')

