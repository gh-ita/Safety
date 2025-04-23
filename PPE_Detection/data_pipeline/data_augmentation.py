import os
import cv2
import shutil
import albumentations as A
from data_merging import extract_imgs
import random

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


def freeze_annot(label_path, clss_prc_dict, seed=42):
    """
    Comment out a percentage of annotations for specific classes in a label file.
    
    Args:
        label_path (str): Path to the YOLO annotation file.
        clss_prc_dict (dict): Dictionary like {class_id: percentage_to_comment_out}.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    new_lines = []
    for class_id in clss_prc_dict:
        clss_lines = [line for line in lines if line.split()[0] == str(class_id)]
        other_lines = [line for line in lines if line.split()[0] != str(class_id)]

        prc = clss_prc_dict[class_id]
        n_to_comment = int(len(clss_lines) * prc)
        lines_to_comment = random.sample(clss_lines, n_to_comment)
        lines_to_keep = [line for line in clss_lines if line not in lines_to_comment]

        # Comment out selected lines
        commented_lines = [f"# {line}" for line in lines_to_comment]

        # Combine everything
        lines = other_lines + lines_to_keep + commented_lines

    lines.sort(key=lambda x: x.replace("// ", "") if x.startswith("#") else x)

    with open(label_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

import os

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


if __name__ == "__main__":
    
    label_list = os.listdir("../data/augmentation data/labels")
    unfreeze_annot("../data/augmentation data/labels")
    """"
    for file in label_list :
        freeze_annot(label_path= os.path.join("../data/augmentation data/labels", file), clss_prc_dict={0:0.3, 9:1})
    """