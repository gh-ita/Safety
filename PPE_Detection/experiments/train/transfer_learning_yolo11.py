from ultralytics import YOLO 


model_checkpoint = "checkpoints/yolo11n.pt"
model = YOLO(model_checkpoint)

#freezing the backbone, neck and bounding box regression layers in the head 
freeze = 23
freeze = [f"model.{x}." for x in range(freeze)] 
freeze.append("model.23.cv2") 
for k, v in model.named_parameters():
    v.requires_grad = True  
    if any(x in k for x in freeze):
        print(f"Freezing layer: {k}")
        v.requires_grad = False
        
#Train the unfrozen classification layers
model.train() 


