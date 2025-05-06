from ultralytics import YOLO 
from sklearn.model_selection import train_test_split 
from ultralytics import YOLO, settings
import mlflow 
import os 


mlflow.set_tracking_uri("file:./mlruns")
settings.update({"mlflow": True})
settings.reset()

MODEL_NAME = "yolo11n_pretrained"
MODEL_CONFIG = "checkpoints/yolo11n.pt"
EXPERIMENTATION_OUTPUT_DIR = "PPE_detection_data_version_3_mask/detect/yolo"
DATA = "../data/augmentation_data/mask/data.yaml"
TRAIN_NAME = "train_yolo11"
TEST_NAME = "train_yolo112"

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
PATIENCE = 30 


model = YOLO(MODEL_CONFIG)

#freezing the backbone, neck and bounding box regression layers in the head 
freeze = 23
freeze = [f"model.{x}." for x in range(freeze)] 
freeze.append("model.23.cv2") 
for k, v in model.named_parameters():
    v.requires_grad = True  
    if any(x in k for x in freeze):
        print(f"Freezing layer: {k}")
        v.requires_grad = False
        
model.train(
            data = DATA,
            epochs = EPOCHS, 
            batch = BATCH, 
            imgsz = IMGSZ, 
            device = DEVICE, 
            profile = PROFILE, 
            project = EXPERIMENTATION_OUTPUT_DIR,
            name = TRAIN_NAME,
            patience = PATIENCE
            )
val_metrics = model.val(data=DATA, split="test",save_json=True)
train_artifact = os.path.join(EXPERIMENTATION_OUTPUT_DIR,TRAIN_NAME)
test_artifact = os.path.join(EXPERIMENTATION_OUTPUT_DIR,TEST_NAME)

#mlflow.log_artifacts(train_artifact, 'training artifact')
#mlflow.log_artifacts(test_artifact, 'test artifact')

print(f"Finished training fold")
print(f"Test mAP50: {val_metrics.box.map50}")


