import xml.etree.ElementTree as ET
import glob
import os
import json
import argparse

def xml_to_yolo_bbox(xmin, ymin, xmax, ymax, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((xmin + xmax) / 2 )/ w
    y_center = ((ymin + ymax) / 2 )/ h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return x_center, y_center, width, height

def convert_voc_to_yolo(input_dir, output_dir, classes_file):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read class names
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split('\n')
    
    # Process each XML file in the input directory
    for xml_file in os.listdir(input_dir):
        if not xml_file.endswith(".xml"):
            continue
        
        tree = ET.parse(os.path.join(input_dir, xml_file))
        root = tree.getroot()
        print(root.findall('object'))
        image_filename = root.find('filename').text
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        yolo_labels = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            
            bbox = obj.find('bndbox')
            
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center, y_center, bbox_width, bbox_height = xml_to_yolo_bbox(x_min, y_min, x_max, y_max, width, height)
            
            yolo_labels.append(f"{cls_id} {x_center} {y_center} {bbox_width} {bbox_height}")
            # print(yolo_labels)
        
        yolo_filename = os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt')
        with open(yolo_filename, 'w') as f:
            f.write('\n'.join(yolo_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLO format.")
    parser.add_argument("--input", type=str, help="Path to the input directory")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--cls", type=str, help="Path to the file containing class names.")
    
    args = parser.parse_args()
    convert_voc_to_yolo(args.input, args.output, args.cls)