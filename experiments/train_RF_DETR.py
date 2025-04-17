import mlflow
from rfdetr import RFDETRBase
import os
import shutil
from datetime import datetime

# === Configuration ===
MODEL_NAME = "RF-DETR Base"  
TEST_DATA_PATH = "../RF-DETR_data/test/data.yaml"           
EXPERIMENT_NAME = "RF-DETR Experiments"
RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"runs/detect/train/{MODEL_NAME}"
MODEL_OUTPUT = f"models/{MODEL_NAME}/"

#hyperparameters 
EPOCHS = 100
BATCH = 16
LR =  1e-4
GRAD_ACCUM_STEPS = 4

# === Start MLflow tracking ===
FOLD_PATHS = [
    "../RF-DETR_data/kfold_base/fold_0/train",
    "../RF-DETR_data/kfold_base/fold_1/train",
    "../RF-DETR_data/kfold_base/fold_2/train",
    "../RF-DETR_data/kfold_base/fold_3/train",
    "../RF-DETR_data/kfold_base/fold_4/train",
]
for fold_idx, fold_data_path in enumerate(FOLD_PATHS):
    RUN_NAME = f"{MODEL_NAME}_fold{fold_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        # Log hyperparameters
        mlflow.log_param("fold", fold_idx)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning rate", LR)
        
        # Train the model
        model = RFDETRBase()
        model.train(dataset_dir=fold_data_path, 
                    epochs=EPOCHS, 
                    batch_size=BATCH,
                    grad_accum_steps=GRAD_ACCUM_STEPS, 
                    lr=LR, 
                    output_dir=OUTPUT_DIR,
                    early_stopping = True,
                    tensorboard = True)
        
        # Copy best model to models/ folder
        best_model_path = os.path.join(OUTPUT_DIR, "weights", "best.pt")
        fold_model_output = f"{MODEL_OUTPUT}_fold{fold_idx}.pt"
        os.makedirs(MODEL_OUTPUT, exist_ok=True)
        shutil.copy(best_model_path, fold_model_output)

        # Log model artifact (raw file)
        mlflow.log_artifact(fold_model_output)

        # Log performance metrics from fold's validation data
        val_metrics = model.val(data=fold_data_path, split='val')
        mlflow.log_metric("val_map50", val_metrics.box.map50)
        mlflow.log_metric("val_map", val_metrics.box.map)
        
        # Log performance metrics from separate test data - ADDED
        test_metrics = model.val(data=TEST_DATA_PATH)
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
if __name__ == "__main__" :
    model = RFDETRBase()
    print(dir(model))
