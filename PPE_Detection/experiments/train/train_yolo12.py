from ultralytics import YOLO, settings
import mlflow 
import os 

mlflow.set_tracking_uri("file:./mlruns")
settings.update({"mlflow": True})
settings.reset()

MODEL_NAME = "yolo12n"
MODEL_CONFIG = "yolo12n.pt"
EXPERIMENTATION_OUTPUT_DIR = "PPE_detection_data_version 1/detect/yolo"
DATA = "data.yaml"
TRAIN_NAME = "train_yolo12"
TEST_NAME = "train_yolo122"

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
model.train(
            data = DATA,
            epochs = EPOCHS, 
            batch = BATCH, 
            imgsz = IMGSZ, 
            device = DEVICE, 
            profile = PROFILE, 
            project = EXPERIMENTATION_OUTPUT_DIR,
            name = TRAIN_NAME)
val_metrics = model.val(data=DATA, split="test",save_json=True)
train_artifact = os.path.join(EXPERIMENTATION_OUTPUT_DIR,TRAIN_NAME)
test_artifact = os.path.join(EXPERIMENTATION_OUTPUT_DIR,TEST_NAME)

mlflow.log_artifacts(train_artifact, 'training artifact')
mlflow.log_artifacts(test_artifact, 'test artifact')

print(f"Finished training fold")
print(f"Test mAP50: {val_metrics.box.map50}")
        

        
        
        
        
        
