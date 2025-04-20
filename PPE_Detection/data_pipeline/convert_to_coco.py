import os
import json
from PIL import Image
import shutil


yolo_labels_path = "splits/kfold_base/fold_1/valid/labels"        
images_path = "splits/kfold_base/fold_1/valid/images"    
sorted_img_list = sorted(os.listdir(images_path))
sorted_lbl_list = sorted(os.listdir(yolo_labels_path))           
output_json = "RF-DETR_data/kfold_base/fold_1/valid/valid_annotations.coco.json"
class_names = ['Gloves', 'Goggles', 'Helmet', 'Mask', 'No-Gloves', 'No-Goggles', 'No-Helmet', 'No-Mask', 'No-Safety_Vest', 'Person', 'Safety_Vest']
source_images_path = "splits/kfold_base/fold_1/valid/images"
destination_images_path = "RF-DETR_data/kfold_base/fold_1/valid"

coco = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

for i, name in enumerate(class_names):
    coco["categories"].append({
        "id": i,
        "name": name,
        "supercategory": "none"
    })

image_id = 0
annotation_id = 0

for index, lbl_filename in enumerate(sorted_lbl_list):
    
    image_filename = sorted_img_list[index]
    image_path = os.path.join(images_path, image_filename)
    label_path = os.path.join(yolo_labels_path, lbl_filename)

    if not os.path.exists(image_path):
        print(f"Skipping {image_filename} — image not found.")
        continue

    # Get image size
    with Image.open(image_path) as img:
        width, height = img.size

    # Add image info
    coco["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": width,
        "height": height
    })

    # Parse YOLO annotations
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_c, y_c, w, h = map(float, parts)
        x_c *= width
        y_c *= height
        w *= width
        h *= height
        x = x_c - w / 2
        y = y_c - h / 2

        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })

        annotation_id += 1

    image_id += 1


with open(output_json, "w") as f:
    json.dump(coco, f, indent=4)

print(f"COCO annotations saved to {output_json}")

"""
for index,lbl_filename in enumerate(sorted_lbl_list):
    if not lbl_filename.endswith(".txt"):
        continue
    
    image_filename = sorted_img_list[index]
    image_path = os.path.join(source_images_path, image_filename)

    if not os.path.exists(image_path):
        print(f"Skipping {image_path} — image not found.")
        continue

    # Copy the image
    destination_image_path = os.path.join(destination_images_path, image_filename)
    shutil.copyfile(image_path, destination_image_path)
"""
