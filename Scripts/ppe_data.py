import os
from PIL import Image

def extract_person_bboxes(annotation_path, class_id=0):
    person_bboxes = []
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            label = int(parts[0])
            if label == class_id:  # Assuming class_id 0 is for 'person'
                bbox = list(map(float, parts[1:]))
                person_bboxes.append(bbox)
    return person_bboxes

def crop_image(image_path, bbox, output_path):
    image = Image.open(image_path)
    width, height = image.size
    x_center, y_center, w, h = bbox
    x_center *= width
    y_center *= height
    w *= width
    h *= height
    left = int(x_center - w / 2)
    top = int(y_center - h / 2)
    right = int(x_center + w / 2)
    bottom = int(y_center + h / 2)
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(output_path)

# Example usage
# crop_image('image.jpg', [0.5, 0.5, 0.5, 0.5], 'cropped_image.jpg')
def adjust_ppe_annotations(annotation_path, person_bbox, output_path):
    x_center, y_center, w, h = person_bbox
    with open(annotation_path, 'r') as infile, open(output_path, 'w') as outfile:
        lines = infile.readlines()
        for line in lines:
            parts = line.strip().split()
            label = int(parts[0])
            if label == 0:
                continue
            bbox = list(map(float, parts[1:]))
            print(f'{output_path} before -- {bbox}')
            # Adjust the coordinates
            bbox[0] = (bbox[0] - x_center + w / 2) / w
            bbox[1] = (bbox[1] - y_center + h / 2) / h
            bbox[2] /= w
            bbox[3] /= h
            print(f'{output_path} after -- {bbox}')
            if 0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1:
                outfile.write(f"{label} {' '.join(map(str, bbox))}\n")

def process_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    for image_file in os.listdir(images_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(images_dir, image_file)
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            
            person_bboxes = extract_person_bboxes(label_path)
            
            for i, person_bbox in enumerate(person_bboxes):
                output_image_path = os.path.join(output_images_dir, f"{os.path.splitext(image_file)[0]}_{i}.jpg")
                output_label_path = os.path.join(output_labels_dir, f"{os.path.splitext(image_file)[0]}_{i}.txt")
                
                crop_image(image_path, person_bbox, output_image_path)
                adjust_ppe_annotations(label_path, person_bbox, output_label_path)

# Example usage
# process_dataset('E:\\INTERNSHIP\\SYOOK\\datasets\\datasets\\images',
#                 'E:\\INTERNSHIP\\SYOOK\\datasets\\datasets\\pytorch_label',
#                 'E:\\INTERNSHIP\\SYOOK\\cropped_data\\train\\images',
#                 'E:\\INTERNSHIP\\SYOOK\\cropped_data\\train\\labels')



