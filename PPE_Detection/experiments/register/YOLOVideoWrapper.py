import mlflow.pyfunc
from ultralytics import YOLO
import supervision as sv
import cv2
import os
import uuid

class YOLOVideoWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = YOLO(context.artifacts["model_path"])
        self.box_annotator = sv.BoxAnnotator()

    def predict(self, context, model_input):
        """
        model_input: dict with 'video_path': path to input video
        Returns: path to annotated video (logged separately as artifact)
        """
        video_path = model_input['video_path']
        output_path = self._annotate_video(video_path)
        return {"annotated_video_path": output_path}

    def _annotate_video(self, input_video_path):
        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        video_id = str(uuid.uuid4())
        output_path = os.path.join(output_dir, f"annotated_{video_id}.mp4")

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            labels = [
                f"{self.model.model.names[class_id]} {conf:.2f}"
                for class_id, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
            out.write(annotated)

        cap.release()
        out.release()
        return output_path
