import cv2
from rfdetr import RFDETRBase
import supervision as sv

LABEL_MAP = ["gloves","goggles","helmet","mask", "no-gloves","no-goggles","no-helmet","no-mask","no-safety-vest","person","safety-vest"]
# Load RFDETR model (change path if you're using a custom one)
model = RFDETRBase(pretrained_weights = "../experiments/checkpoints/RF-DETR.pth")

# Input and output paths
input_path = "../../test_video_trimmed.mp4"
output_path = "annotated_output_rf-detr.mp4"

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
label_annotator = sv.LabelAnnotator(text_scale=0.3, text_thickness=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run RFDETR inference
    detections = model.predict(frame, threshold = 0.5)

    # Generate labels (class name + confidence)
    labels = [
        f"{LABEL_MAP[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
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
