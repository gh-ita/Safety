import mlflow
import os
import shutil
import mlflow.tracking
from datetime import datetime
from ultralytics import YOLO

# === Configuration ===
MODEL_NAME = "yolov12n"
CONFIG_PATH = "yolo12n.pt"
TEST_DATA_PATH = "../splits/test/data.yaml"
EXPERIMENT_NAME = "YOLOv12 Experiments"
MODEL_OUTPUT_DIR = f"models/{MODEL_NAME}/"
EPOCHS = 100
BATCH = 16
IMGSZ = 640
LR = None
PROFILE = True
FREEZE = 0
DROPOUT = 0
WEIGHT_DECAY = 0.0005
DEVICE = 0

FOLD_PATHS = [
    "../splits/kfold_base/fold_0/data.yaml",
    "../splits/kfold_base/fold_1/data.yaml",
    "../splits/kfold_base/fold_2/data.yaml",
    "../splits/kfold_base/fold_3/data.yaml",
    "../splits/kfold_base/fold_4/data.yaml",
]

mlflow.set_tracking_uri("file:./mlruns")  # safer
mlflow.set_experiment(EXPERIMENT_NAME)

for fold_idx, fold_data_path in enumerate(FOLD_PATHS):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{MODEL_NAME}_fold{fold_idx}_{timestamp}"
    output_dir = f"runs/detect/{MODEL_NAME}_fold{fold_idx}_{timestamp}"
    fold_model_output = f"{MODEL_OUTPUT_DIR}_fold{fold_idx}.pt"
    
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Log hyperparameters
            mlflow.log_param("fold", fold_idx)
            mlflow.log_param("layers_frozen", FREEZE)
            mlflow.log_param("image_size", IMGSZ)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("learning_rate", LR)
            mlflow.log_param("dropout", DROPOUT)
            mlflow.log_param("weight_decay", WEIGHT_DECAY)

            # Train model
            model = YOLO(CONFIG_PATH)
            model.train(
                data=fold_data_path,
                epochs=EPOCHS,
                batch=BATCH,
                profile=PROFILE,
                device=DEVICE,
                project=output_dir
            )

            # Save best model
            best_model_path = os.path.join(output_dir, "weights", "best.pt")
            os.makedirs("models", exist_ok=True)
            shutil.copy(best_model_path, fold_model_output)

            # Log model file as artifact
            mlflow.log_artifact(fold_model_output)

            # Validation on val split
            val_metrics = model.val(data=fold_data_path, split='val')
            mlflow.log_metric("val_map50", val_metrics.box.map50)
            mlflow.log_metric("val_map", val_metrics.box.map)

            # Test set evaluation
            test_metrics = model.val(data=TEST_DATA_PATH, split='test')
            mlflow.log_metric("test_map50", test_metrics.box.map50)
            mlflow.log_metric("test_map", test_metrics.box.map)

            print(f"✅ Fold {fold_idx} completed.")
            print(f"📈 Val mAP50: {val_metrics.box.map50}, Test mAP50: {test_metrics.box.map50}")

            # MLflow Registry
            mlflow.log_artifact(fold_model_output, artifact_path="model")
            model_uri = f"runs:/{run.info.run_id}/model"
            model_name_registry = f"{MODEL_NAME}_model"

            client = mlflow.tracking.MlflowClient()
            result = mlflow.register_model(model_uri, model_name_registry)
            client.transition_model_version_stage(
                name=model_name_registry,
                version=result.version,
                stage="Staging"
            )
            print(f"🚀 Model for fold {fold_idx} registered and moved to 'Staging'.")

        except Exception as e:
            print(f"❌ Fold {fold_idx} failed: {e}")
            mlflow.log_param("crash_reason", str(e))
            raise
