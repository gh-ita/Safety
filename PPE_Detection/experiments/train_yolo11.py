from ultralytics import YOLO 
from datetime import datetime
import mlflow 
import os 

#metrics 
#artifact

MODEL_NAME = "yolo11n"
MODEL_CONFIG = "yolo11n.pt"
EXPERIMENTATION = "PPE detection data version 1"
RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')} trained on 5 folds"
OUTPUT_DIR = f"runs/detect/{MODEL_NAME}"
DATA = "../splits/kfold_base/fold_0/data.yaml"
NAME = "train"

#hyperparameters
EPOCHS = 100
BATCH = 16
IMGSZ = 640
LR =  None
PROFILE = True 
FREEZE = 0 #default
DROPOUT = 0 #default
WEIGHT_DECAY = 0.0005 #default
DEVICE = 0


model = YOLO(MODEL_CONFIG)
mlflow.set_experiment(EXPERIMENTATION)
with mlflow.start_run(run_name = RUN_NAME) as run :
    #log hyperparameters 
    mlflow.log_param("Epochs",EPOCHS)
    mlflow.log_param("Batch",BATCH)
    mlflow.log_param("Image size",IMGSZ)
    mlflow.log_param("Layers freezed",IMGSZ)
    mlflow.log_param("Dropout",DROPOUT)
    mlflow.log_param("Weight decay",WEIGHT_DECAY)
    
    model.train(
                data = DATA,
                epochs = EPOCHS, 
                batch = BATCH, 
                imgsz = IMGSZ, 
                device = DEVICE, 
                profile = PROFILE, 
                project = OUTPUT_DIR,
                name = NAME)
    
    artifact = os.path.join(OUTPUT_DIR,NAME,"weights","best.pt")
    #log model artifact
    mlflow.log_artifact(artifact)
    
    #log evaluation metrics 
    val_metrics = model.val(data = DATA, split= "test")
    mlflow.log_metric("val_map50", val_metrics.box.map50)
    mlflow.log_metric("val_map", val_metrics.box.map)
    
    print(f"Finished training fold")
    print(f"Test mAP50: {val_metrics.box.map50}")
    

    
    
    
    
    
