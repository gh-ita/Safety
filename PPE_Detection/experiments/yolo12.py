from ultralytics import YOLO 
from datetime import datetime
import mlflow 
import os 
import numpy as np

os.environ("MLFLOW_TRACKING_URI") = "custom"

#metrics 
#artifact

MODEL_NAME = "yolo12n"
MODEL_CONFIG = "yolo12n.pt"
EXPERIMENTATION = "PPE detection data version 1"
RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')} trained on 5 folds"
OUTPUT_DIR = f"runs/detect/{MODEL_NAME}"
TRAIN_NAME = "train_"
TEST_NAME = "train_62"
FOLDS = 5

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

box_losses = []
cls_losses = []
mAP_50s = []
mAP_50_95s = []

mlflow.set_experiment(EXPERIMENTATION)
with mlflow.start_run(run_name = RUN_NAME) as run :
    #log hyperparameters 
    mlflow.log_param("Epochs",EPOCHS)
    mlflow.log_param("Batch",BATCH)
    mlflow.log_param("Image size",IMGSZ)
    mlflow.log_param("Layers freezed",IMGSZ)
    mlflow.log_param("Dropout",DROPOUT)
    mlflow.log_param("Weight decay",WEIGHT_DECAY)
        
for i in range(FOLDS):
    DATA = f"../splits/kfold_base/fold_{i}/data.yaml"
    NAME = f"train_"
    model = YOLO(MODEL_CONFIG)
    model.train(
                data = DATA,
                epochs = EPOCHS, 
                batch = BATCH, 
                imgsz = IMGSZ, 
                device = DEVICE, 
                profile = PROFILE, 
                project = OUTPUT_DIR,
                name = NAME)
    
    # Evaluate on test split
    val_metrics = model.val(data=DATA, split="test",save_json=True)
    # Collect metrics (use .results if required by your YOLO version)
    mAP_50s.append(val_metrics.box.map50)
    mAP_50_95s.append(val_metrics.box.map)

avg_box_loss = np.mean(box_losses)
avg_cls_loss = np.mean(cls_losses)
avg_mAP_50 = np.mean(mAP_50s)
avg_mAP_50_95 = np.mean(mAP_50_95s)

#check the structure of the results object 
#copy the test folder in each fold folder

train_artifact = os.path.join(OUTPUT_DIR,TRAIN_NAME)
test_artifact = os.path.join(OUTPUT_DIR,TEST_NAME)

#log model artifact and result artifacts 
mlflow.log_artifacts(train_artifact, 'training artifact')
mlflow.log_artifacts(test_artifact, 'test artifact')

#log evaluation metrics 
mlflow.log_metric("Avg Box Loss", float(avg_box_loss))
mlflow.log_metric("Avg Cls Loss", float(avg_cls_loss))
mlflow.log_metric("Avg mAP@0.5", float(avg_mAP_50))
mlflow.log_metric("Avg mAP@0.5:0.95", float(avg_mAP_50_95))

print(f"Finished training fold")
print(f"Test mAP50: {val_metrics.box.map50}")

    

    
    
    
    
    
