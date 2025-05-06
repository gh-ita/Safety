import cv2
from ultralytics import YOLO
import supervision as sv

# Load model
model = YOLO("../experiments/checkpoints/yolo11n.pt")

# Input and output image paths
input_image_path = "C:/Users/ghita/Downloads/googles.png"
output_image_path = "googles_yolo11n_annotated_sample_image.png"

# Load image
image = cv2.imread(input_image_path)

# Inference
results = model(image)[0]

# Convert results to Supervision Detections
detections = sv.Detections.from_ultralytics(results)

# Optional: filter detections by confidence
detections = detections[detections.confidence > 0.3]

# Prepare labels (class name + confidence)
labels = [
    f"{model.model.names[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

# Annotate image
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.3)

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# Save the annotated image
cv2.imwrite(output_image_path, annotated_image)
print(f"Saved annotated image to: {output_image_path}")
