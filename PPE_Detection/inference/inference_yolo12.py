import cv2
from ultralytics import YOLO
import supervision as sv

# Load YOLOv12n model (change path if you're using a custom one)
model = YOLO("../experiments/checkpoints/yolo12n.pt")

# Input and output paths
input_path = "../../goggle_video.mp4"
output_path = "annotated_goggles_output_yolo12n.mp4"

# Open input video
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Supervision annotator (set text and box styles here if needed)
annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv12n inference
    results = model(frame)[0]

    # Convert to Supervision detections
    detections = sv.Detections.from_ultralytics(results)

    # Optional: filter low-confidence detections
    detections = detections[detections.confidence > 0.3]

    # Generate labels (class name + confidence)
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate frame
    annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections)
    # Optional: add global label (e.g., "PPE Detection") on top of the same frame
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections = detections)
    # Write final annotated frame
    writer.write(annotated_frame)



# Cleanup
cap.release()
writer.release()
