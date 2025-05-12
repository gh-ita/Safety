from rfdetr import RFDETRBase
import mlflow 
import supervision as sv
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision
from PIL import Image
import os
import matplotlib.pyplot as plt

DATA = "../dataset"
EPOCHS = 100
BATCH = 4
LR = 1e-4
GRAD_ACCUM_STEPS = 4
EXPERIMENTATION = "PPE_detection_data_version 1/detect/yolo"
TRAIN_NAME = "train_RF_DETR_BASE"
TEST_NAME =  "train_RF_DETR_BASE2"
OUTPUT_DIR = "PPE_detection_data_version 1/detect/yolo/train_RF_DETR_BASE"
EARLY_STOPPING = True

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENTATION)

with mlflow.start_run(run_name = TRAIN_NAME) as run :
    mlflow.log_param("Epochs", EPOCHS)
    mlflow.log_param("Batch", BATCH)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("Gradient_accumulation_steps", GRAD_ACCUM_STEPS)
    model = RFDETRBase()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.train(dataset_dir = DATA, 
                epochs = EPOCHS, 
                batch_size = BATCH,
                grad_accum_steps = GRAD_ACCUM_STEPS,
                lr = LR,
                early_stopping = EARLY_STOPPING,
                output_dir = OUTPUT_DIR)
    
    targets = []
    predictions = []
    ds = sv.DetectionDataset.from_coco(
        images_directory_path = f"{DATA}/test" ,
        annotations_path = f"{DATA}/test/annotations.coco.json" ,
    )
    
    for path, image, annotations in tqdm(ds):
        image = Image.open(path)
        detections = model.predict(image, threshold = 0.5)
        
        targets.append(annotations)
        predictions.append(detections)
    
    #compute mAP
    map_metric = MeanAveragePrecision()
    map_result = map_metric.update(predictions, targets).compute()
    mlflow.log_metric("test_mAP", map_result["mAP"])
    
    #plot mAP
    map_fig = map_result.plot()
    map_plot_path = os.path.join(OUTPUT_DIR, "map_plot.png")
    map_fig.savefig(map_plot_path)
    plt.close(map_fig)
    
    #compute confusion matrix
    confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=predictions,
    targets=targets,
    classes=ds.classes
    )
    
    #plot confusion matrix
    cm_fig = confusion_matrix.plot()
    cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    cm_fig.savefig(cm_plot_path)
    plt.close(cm_fig)
    
    #log artifacts
    mlflow.log_artifacts(OUTPUT_DIR, 'training_artifact')