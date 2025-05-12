import cv2
import argparse
from ultralytics import YOLO

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLOv8 Live Webcam Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLOv8 model (e.g., yolov8n.pt)")
    args = parser.parse_args()

    # Load the YOLO model
    model = YOLO(args.model)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running YOLOv8 live inference... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference and annotate frame
        results = model(frame)[0]
        annotated_frame = results.plot()

        # Show the frame
        cv2.imshow("YOLOv8 Live Inference", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
