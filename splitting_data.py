import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(images_dir, labels_dir, output_dir, split_ratio=0.8):
    # Create output directories for train and val sets
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Split the dataset
    train_images, val_images = train_test_split(images, test_size=1 - split_ratio, random_state=42)
    
    def copy_files(images, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
        for image in images:
            label = os.path.splitext(image)[0] + '.txt'
            shutil.copy(os.path.join(src_images_dir, image), os.path.join(dst_images_dir, image))
            shutil.copy(os.path.join(src_labels_dir, label), os.path.join(dst_labels_dir, label))
    
    # Copy train files
    copy_files(train_images, images_dir, labels_dir, train_images_dir, train_labels_dir)
    
    # Copy val files
    copy_files(val_images, images_dir, labels_dir, val_images_dir, val_labels_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument("--images_dir", type=str, help="Path to the directory containing images.")
    parser.add_argument("--labels_dir", type=str, help="Path to the directory containing YOLO labels.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory where train/val sets will be saved.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio of training set size to the whole dataset. Default is 0.8.")
    
    args = parser.parse_args()
    split_dataset(args.images_dir, args.labels_dir, args.output_dir, args.split_ratio)
