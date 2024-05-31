import os
import cv2
from ultralytics import YOLO

def load_model(model_path):
    """Load a YOLO model from the given path."""
    print(model_path)
    model = YOLO(model_path)  # pretrained YOLOv8n model
    return model

def perform_inference(model, image_path, confidence):
    """Perform inference using the given model and image."""
    
    results = model(image_path, conf = confidence)
    return results

def draw_predictions(image, boxes, color, label_map):
    """Draw bounding boxes and labels on the image using OpenCV."""
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
        conf = float(box.conf[0].cpu().numpy())  # Get confidence score
        cls = int(box.cls[0].cpu().numpy())  # Get class label

        # Draw the bounding box
        cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)

        # Put the label and confidence score
        label_text = f"{label_map.get(cls, 'Class')}: {conf:.2f}"
        cv2.putText(image, label_text, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_images(input_dir, output_dir, person_model, ppe_model, person_label_map, ppe_label_map, confidence):
    """Process images from the input directory, perform inference, and save results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_file in os.listdir(input_dir):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, image_file)
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Perform inference using both models
            results_person = perform_inference(person_model, image_path, confidence)
            results_ppe = perform_inference(ppe_model, image_path, confidence)

            # Draw predictions from both models
            draw_predictions(image, results_person[0].boxes, (0, 255, 0), person_label_map)  # Green for person
            draw_predictions(image, results_ppe[0].boxes, (255, 0, 0), ppe_label_map)       # Blue for PPE
            
            # Save the image with drawn predictions
            cv2.imwrite(output_path, image)
            print(f"Processed {image_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference and draw bounding boxes on images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save results.")
    parser.add_argument("--person_model", type=str, required=True, help="Path to the person detection model.")
    parser.add_argument("--ppe_model", type=str, required=True, help="Path to the PPE detection model.")
    parser.add_argument("--conf", type=float, default=0.5, help="Set the confidence Value")
    
    args = parser.parse_args()
    
    person_model = load_model(args.person_model)
    ppe_model = load_model(args.ppe_model)
    
    # Define label maps for person and PPE models
    person_label_map = {0: 'person'}
    ppe_label_map = {1: 'hard-hat', 2: 'gloves', 3: 'boots', 4: 'vest', 5: 'ppe-suit'}
    
    process_images(args.input_dir, args.output_dir, person_model, ppe_model, person_label_map, ppe_label_map, args.conf)
