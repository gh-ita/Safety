import mlflow

EPOCHS = 100
BATCH = 4
LR = 1e-4
GRAD_ACCUM_STEPS = 4

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("PPE_detection_data_version 1/detect/yolo")
with mlflow.start_run(run_name = "train_rf_detr_base") as run :
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch", BATCH)
    mlflow.log_param("lr0", LR)
    mlflow.log_param("Gradient_accumulation_steps", GRAD_ACCUM_STEPS)
    
    mlflow.log_artifacts("PPE_detection_data_version 1/detect/yolo/train_rf_detr_base", "training_artifact")
    mlflow.log_metric("metrics/mAP50B", 0.7609446291051625)
    mlflow.log_metric("metrics/mAP95B", 0.46410041794945417)
    mlflow.log_metric("metrics/recallB", 0.29)