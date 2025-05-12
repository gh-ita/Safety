import mlflow
from YOLOVideoWrapper import YOLOVideoWrapper

artifacts = {"model_path": "../checkpoints/yolo11n.pt"}
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("PPE_detection_data_version 1/detect/yolo")

with mlflow.start_run(run_name="YOLO11n_Model_Registration"):
    mlflow.pyfunc.log_model(
        artifact_path="yolo_video_model",
        python_model=YOLOVideoWrapper(),
        artifacts=artifacts,
        registered_model_name="YOLO11n"
    )
