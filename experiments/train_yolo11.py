import mlflow
import os
import shutil
from datetime import datetime
from ultralytics import YOLO

# === Configuration ===
MODEL_NAME = "yolov11n"   
CONFIG_PATH = "yolo11n.pt"
DATA_PATH = "../Construction-Site-Safety/data.yaml"           
EPOCHS = 50
EXPERIMENT_NAME = "YOLOv11 Experiments"
RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = "runs/detect/train"
MODEL_OUTPUT = "models/yolo11/n"
#hyperparameters 


# === Start MLflow tracking ===
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=RUN_NAME) as run:
    # Log hyperparameters
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("epochs", EPOCHS)

    # Train the model
    model = YOLO(CONFIG_PATH)
    model.train(data=DATA_PATH, epochs=EPOCHS)

    # Copy best model to models/ folder
    best_model_path = os.path.join(OUTPUT_DIR, "weights", "best.pt")
    os.makedirs("models", exist_ok=True)
    shutil.copy(best_model_path, MODEL_OUTPUT)

    # Log model artifact (raw file)
    mlflow.log_artifact(MODEL_OUTPUT)

    # Optional: log performance metrics (requires you to extract them from validation)
    metrics = model.val()
    mlflow.log_metric("map50", metrics.box.map50)
    mlflow.log_metric("map", metrics.box.map)

    print(f"Finished training and tracking: {MODEL_OUTPUT}")

    # === MLflow Model Registry Integration ===
    # Log the model to MLflow's registry
    mlflow.pytorch.log_model(model, "model")

    # Register the model to the registry
    model_uri = f"runs:/{run.info.run_id}/model"
    model_name_registry = f"{MODEL_NAME}_model" 

    # Register the model
    mlflow.register_model(model_uri, model_name_registry)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name_registry,
        version=1,
        stage="Staging"
    )

    print(f"Model registered and transitioned to 'Staging' stage.")