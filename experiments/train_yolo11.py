import mlflow
import mlflow.tracking
import os
import shutil
from datetime import datetime
from ultralytics import YOLO

# === Configuration ===
MODEL_NAME = "yolov11n"   
CONFIG_PATH = "yolo11n.pt"
#DATA_PATH = "../Construction-Site-Safety/data.yaml"
TEST_DATA_PATH = "../splits/test/data.yaml"           
EXPERIMENT_NAME = "YOLOv11 Experiments"
RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"runs/detect/{MODEL_NAME}"
MODEL_OUTPUT = f"models/{MODEL_NAME}/"
#hyperparameters 
EPOCHS = 100
BATCH = 16
IMGSZ = 640
LR =  None
PROFILE = True 
FREEZE = 0
DROPOUT = 0
WEIGHT_DECAY = 0.0005
DEVICE = 0

# === Start MLflow tracking ===
FOLD_PATHS = [
    "../splits/kfold_base/fold_0/data.yaml",
    "../splits/kfold_base/fold_1/data.yaml",
    "../splits/kfold_base/fold_2/data.yaml",
    "../splits/kfold_base/fold_3/data.yaml",
    "../splits/kfold_base/fold_4/data.yaml",
]

for fold_idx, fold_data_path in enumerate(FOLD_PATHS):
    RUN_NAME = f"{MODEL_NAME}_fold{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(EXPERIMENT_NAME)
    try:
        with mlflow.start_run(run_name=RUN_NAME) as run:
            # Log hyperparameters
            mlflow.log_param("fold", fold_idx)
            mlflow.log_param("layers frozen", FREEZE)       
            mlflow.log_param("image size", IMGSZ)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("learning rate", LR)
            mlflow.log_param("dropout", DROPOUT)
            mlflow.log_param("weight decay",WEIGHT_DECAY)
            # Train the model
            model = YOLO(CONFIG_PATH)
            model.train(data=fold_data_path, epochs=EPOCHS, batch = BATCH, profile = PROFILE, device = DEVICE, project = OUTPUT_DIR )

            # Copy best model to models/ folder
            best_model_path = os.path.join(OUTPUT_DIR, "weights", "best.pt")
            fold_model_output = f"{MODEL_OUTPUT}_fold{fold_idx}.pt"
            os.makedirs("models", exist_ok=True)
            shutil.copy(best_model_path, fold_model_output)

            # Log model artifact (raw file)
            mlflow.log_artifact(fold_model_output)

            # Log performance metrics from fold's validation data
            val_metrics = model.val(data=fold_data_path, split='val')
            mlflow.log_metric("val_map50", val_metrics.box.map50)
            mlflow.log_metric("val_map", val_metrics.box.map)
            
            # Log performance metrics from separate test data - ADDED
            test_metrics = model.val(data=TEST_DATA_PATH,split='test')
            mlflow.log_metric("test_map50", test_metrics.box.map50)
            mlflow.log_metric("test_map", test_metrics.box.map)
            
            print(f"Finished training fold {fold_idx}")
            print(f"Validation mAP50: {val_metrics.box.map50}")
            print(f"Test mAP50: {test_metrics.box.map50}")

            # === MLflow Model Registry Integration ===
            # Log the model to MLflow's registry
            mlflow.pytorch.log_model(model, "model")

            # Register the model to the registry
            model_uri = f"runs:/{run.info.run_id}/model"
            model_name_registry = f"{MODEL_NAME}_model" 

            # Register the model
            client = mlflow.tracking.MlflowClient()
            result = mlflow.register_model(model_uri, model_name_registry)
            client.transition_model_version_stage(
                name=model_name_registry,
                version=result.version,
                stage="Staging"
            )

            print(f"Model registered and transitioned to 'Staging' stage.")
    except Exception as e:
        print("Exception during training:", e)
        # Avoid trying to end a run that may not have started
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
