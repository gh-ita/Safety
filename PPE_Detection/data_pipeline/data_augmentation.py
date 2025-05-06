import os
import cv2
import shutil
from data_merging import extract_imgs
import random
import albumentations as A
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, BboxParams
)
from glob import glob

# Paths
IMG_FOLDER = '../data/data/images'
LBL_FOLDER = '../data/data/labels'
SAVE_IMG_FOLDER = '../data/data/augmented_negatives/images'
SAVE_LBL_FOLDER = '../data/data/augmented_negatives/labels'

os.makedirs(SAVE_IMG_FOLDER, exist_ok=True)
os.makedirs(SAVE_LBL_FOLDER, exist_ok=True)

# Strong augmentations
strong_neg_augment = A.Compose([
    A.RandomResizedCrop(size=(640, 640), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.MotionBlur(p=0.2),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.CoarseDropout(p=0.3)

])



def freeze_annot(label_path, clss_prc_dict, seed=42, output_path=None):
    """
    Remove a percentage of annotations for specific classes in a label file.
    
    Args:
        label_path (str): Path to the YOLO annotation file.
        clss_prc_dict (dict): Dictionary like {class_id: percentage_to_remove}.
        seed (int): Random seed for reproducibility.
        output_path (str): Optional path to save the modified file. If None, overwrite the original.
    """
    random.seed(seed)
    
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    new_lines = []
    for class_id in clss_prc_dict:
        # Get lines for the current class
        clss_lines = [line for line in lines if line.split()[0] == str(class_id)]
        other_lines = [line for line in lines if line.split()[0] != str(class_id)]

        prc = clss_prc_dict[class_id]
        n_to_remove = int(len(clss_lines) * prc)
        lines_to_remove = random.sample(clss_lines, n_to_remove)

        # Remove the selected lines
        clss_lines = [line for line in clss_lines if line not in lines_to_remove]

        # Add the remaining class lines and other lines to new_lines
        new_lines.extend(other_lines + clss_lines)

    # Decide where to save the modified content
    save_path = output_path if output_path else label_path

    # Save the modified content back to the file
    with open(save_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')



def unfreeze_annot(label_path):
    for fname in os.listdir(label_path):
        if not fname.endswith(".txt"):
            continue

        file_path = os.path.join(label_path, fname)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # Remove first character (comment marker)
                line = stripped[1:]
            new_lines.append(line)

        with open(file_path, 'w') as f:
            f.writelines(new_lines)




transform = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    RandomBrightnessContrast(p=0.2),
],
bbox_params=BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.0,
    check_each_transform=False,
    clip=True  
))

def augment_image(image_path, label_path, save_dir_img, save_dir_lbl, n_aug=2):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []

    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        bbox = list(map(float, parts[1:]))
        bboxes.append(bbox)
        class_labels.append(cls)

    for i in range(n_aug):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # Save augmented image
        aug_img_name = os.path.splitext(os.path.basename(image_path))[0] + f"_aug{i}.jpg"
        cv2.imwrite(os.path.join(save_dir_img, aug_img_name), aug_img)

        # Save corresponding label
        aug_lbl_name = aug_img_name.replace('.jpg', '.txt')
        with open(os.path.join(save_dir_lbl, aug_lbl_name), 'w') as f:
            for lbl, box in zip(aug_labels, aug_bboxes):
                f.write(f"{lbl} {' '.join(map(str, box))}\n")


if __name__ == "__main__":
    image_dir = "../data/merged_data/no-gloves/images"
    label_dir = "../data/merged_data/no-gloves/labels"
    save_dir_img = "../data/merged_data/augmented_data_2/no-gloves/images"
    save_dir_lbl = "../data/merged_data/augmented_data_2/no-gloves/labels"

    # Loop through each image file
    for img_file in os.listdir(image_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files
        
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

        augment_image(
            image_path=image_path,
            label_path=label_path,
            save_dir_img=save_dir_img,
            save_dir_lbl=save_dir_lbl
        )
    """
    label_list = os.listdir("../data/augmentation data/labels")

    for file in label_list :
        #freeze_annot(label_path= os.path.join("../data/augmentation data/labels", file), clss_prc_dict={0:0.3, 9:1})
        freeze_annot(label_path=os.path.join("../data/augmentation data/labels", file), clss_prc_dict={0: 0.3, 9: 0.5}, output_path=os.path.join("../data/augmentation data/aug_labels", file))
    """