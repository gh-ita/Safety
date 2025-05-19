import time
import cv2
from ultralytics import YOLO

""" # Load the YOLO11 model
model = YOLO("/home/sysadmin01/Desktop/PFE/safety/PPE_Detection/experiments/checkpoints/yolo11n_v6.pt")

# Export the model to TensorRT format
model.export(format="engine", device = 0, nms = True)  # creates 'yolo11n.engine' """


# Load TensorRT model
model = YOLO("/home/sysadmin01/Desktop/PFE/safety/PPE_Detection/experiments/checkpoints/yolo11n_v6.engine")

# Open webcam
cap = cv2.VideoCapture(0)

frame_count = 0
total_inference_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Measure inference time
    start_time = time.time()
    results = model(frame, conf=0.5, device=0)
    inference_time = time.time() - start_time

    # Update stats
    frame_count += 1
    total_inference_time += inference_time

    # Show frame with detections
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO + TensorRT", annotated_frame)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break


# Print average FPS
average_fps = frame_count / total_inference_time
print(f"Processed {frame_count} frames")
print(f"Average FPS: {average_fps:.2f}")
print(f"Average inference time per frame: {1000 / average_fps:.2f} ms")

cap.release()
cv2.destroyAllWindows()
