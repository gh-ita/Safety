#Stratified k-fold splitting 
import os 
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import shutil
import numpy as np 
from collections import Counter 
import random
import shutil
from pathlib import Path

LBL_DIR_PATH = "../data/merged_data/augmented_data_2/labels"
SORTED_LBL_LIST = sorted(os.listdir(LBL_DIR_PATH))
IMG_DIR_PATH = "../data/merged_data/augmented_data_2/images"
SORTED_IMG_LIST = sorted(os.listdir(IMG_DIR_PATH))

def get_proxy_lbl(sorted_lbl_list, 
                  lbl_folder_path):
    """
    A method that return a list of proxy labels for the image dataset
    The proxy label is chosen as the most redundant obj in each image
    """
    proxy_lbls = [0 for _ in range(len(sorted_lbl_list))]
    for index, lbl_file in enumerate(sorted_lbl_list):
        lbl_file_path = os.path.join(lbl_folder_path,lbl_file)
        obj_lst = [0 for _ in range(11)]
        with open(lbl_file_path, "r") as file :
            for line in file :
                obj_lst[int(line.split()[0])] += 1
        proxy_lbls[index] = obj_lst.index(max(obj_lst))
    return proxy_lbls

def filter_rare_classes(images, 
                        labels, 
                        proxy_labels, 
                        images_dir, 
                        labels_dir, 
                        min_count=2, 
                        output_dir="rare_classes"):
    """
    Filters out samples whose proxy label appears less than `min_count` times.
    Saves the rare class images and labels into a separate folder.
    """
    valid_img_dir = "../data/augmentation_data/mask/filtered_data/images"
    valid_lbl_dir = "../data/augmentation_data/mask/filtered_data/labels"
    counts = Counter(proxy_labels)
    rare_indices = [i for i, lbl in enumerate(proxy_labels) if counts[lbl] < min_count]
    rare_img_dir = os.path.join(output_dir, "images")
    rare_lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(rare_img_dir, exist_ok=True)
    os.makedirs(rare_lbl_dir, exist_ok=True)
    
    for idx in rare_indices:
        img_name = images[idx]
        lbl_name = labels[idx]
        shutil.copy(os.path.join(images_dir, img_name), os.path.join(rare_img_dir, img_name))
        shutil.copy(os.path.join(labels_dir, lbl_name), os.path.join(rare_lbl_dir, lbl_name))
    print(f"Filtered {len(rare_indices)} rare class samples into {output_dir}.")
    

        
    valid_indices = [i for i in range(len(proxy_labels)) if counts[proxy_labels[i]] >= min_count]
    valid_images = [images[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    valid_proxy_labels = [proxy_labels[i] for i in valid_indices]
    
    for idx in valid_indices:
        img_name = images[idx]
        lbl_name = labels[idx]
        shutil.copy(os.path.join(images_dir, img_name), os.path.join(valid_img_dir, img_name))
        shutil.copy(os.path.join(labels_dir, lbl_name), os.path.join(valid_lbl_dir, lbl_name))
    return valid_images, valid_labels, valid_proxy_labels

    
def generate_folds(n_splits,
                   sorted_img_list,
                   sorted_lbl_list,
                   proxy_lbls,
                   images_dir, 
                   labels_dir, 
                   output_dir):
    """
    A method that generates the K folds using StratifiedKFold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)              
    for fold, (train_idx, val_idx) in enumerate(skf.split(sorted_img_list, proxy_lbls)):
        print(f"\n🔁 Fold {fold + 1}")
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        
        for split in ['train', 'valid']:
            os.makedirs(os.path.join(fold_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, split, 'labels'), exist_ok=True)

        # Create a mapping between index lists and folder names
        for idxs, split in [(train_idx, 'train'), (val_idx, 'valid')]:
            for img_name, lbl_name in zip(np.array(sorted_img_list)[idxs], np.array(sorted_lbl_list)[idxs]):
                # Define source paths
                src_img = os.path.join(images_dir, img_name)
                src_lbl = os.path.join(labels_dir, lbl_name)

                # Define destination paths (nested structure)
                dst_img = os.path.join(fold_dir, split, 'images', img_name)
                dst_lbl = os.path.join(fold_dir, split, 'labels', lbl_name)

                # Copy files
                shutil.copyfile(src_img, dst_img)
                shutil.copyfile(src_lbl, dst_lbl)

        print("Train:", len(train_idx), "Validation:", len(val_idx), "copied successfully.")


def split_yolo_dataset(
    base_path,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42
):
    random.seed(seed)

    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")

    # Get only image files that have corresponding labels
    img_files = [f for f in os.listdir(images_path)
                 if True and
                 os.path.exists(os.path.join(labels_path, os.path.splitext(f)[0] + ".txt"))]

    random.shuffle(img_files)
    total = len(img_files)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": img_files[:train_end],
        "val": img_files[train_end:val_end],
        "test": img_files[val_end:]
    }

    for split, files in splits.items():
        for split_type in ["images", "labels"]:
            split_dir = os.path.join(base_path, split, split_type)
            os.makedirs(split_dir, exist_ok=True)

        for img_file in files:
            label_file = os.path.splitext(img_file)[0] + ".txt"

            # Full paths for source files
            src_img_path = os.path.join(images_path, img_file)
            src_label_path = os.path.join(labels_path, label_file)

            # Full paths for destination files
            dest_img_path = os.path.join(base_path, split, "images", img_file)
            dest_label_path = os.path.join(base_path, split, "labels", label_file)

            shutil.copy(src_img_path, dest_img_path)
            shutil.copy(src_label_path, dest_label_path)

    print(f"✅ Split complete! Total: {total} -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


def select_class_data(clss_id_list, img_folder, lbl_folder):
    clss_img_list = []
    clss_lbl_list = []
    clss_id_set = set(clss_id_list)
    
    img_list = sorted(os.listdir(img_folder))
    lbl_list = sorted(os.listdir(lbl_folder))
    
    for index, lbl_file in enumerate(lbl_list):
        lbl_path = os.path.join(lbl_folder, lbl_file)
        with open(lbl_path, "r") as lbls:
            for line in lbls:
                if not line :
                    continue
                try:
                    clss = int(line.split()[0])
                    if clss in clss_id_set:
                        clss_img_list.append(img_list[index])
                        clss_lbl_list.append(lbl_file)
                        break 
                except (ValueError, IndexError):
                    continue 
    return clss_img_list, clss_lbl_list


def remove_annot(clss_id_list, lbl_folder):
    clss_id_set = set(clss_id_list)  
    lbl_list = os.listdir(lbl_folder)

    for lbl_file in lbl_list:
        file_new_lbl = []
        lbl_path = os.path.join(lbl_folder, lbl_file)
        
        with open(lbl_path, "r") as lbls:
            for line in lbls:
                line = line.strip()
                if not line:
                    continue
                try:
                    clss = int(line.split()[0])
                    if clss not in clss_id_set:
                        file_new_lbl.append(line + "\n")
                except (ValueError, IndexError):
                    continue
        
        with open(lbl_path, "w") as lbls:
            lbls.writelines(file_new_lbl)                                  
    
    
    
if __name__ == "__main__":
    #negatives [4,5,6,7,8,9]
    #positives [0,1,2,3,9,10]
    dest_img_folder = "../data/merged_data/augmented_data_2/p_clss/images"
    dest_lbl_folder = "../data/merged_data/augmented_data_2/n_clss/labels"
    #n
    remove_annot([0,1,2,3,10], dest_lbl_folder)
    
    """
    img_list, lbl_list = select_class_data([0,1,2,3,9,10], IMG_DIR_PATH, LBL_DIR_PATH)
    
    for lbl in lbl_list :
        src_lbl_path = os.path.join(LBL_DIR_PATH, lbl)
        dest_lbl_path = os.path.join(dest_lbl_folder, lbl)
        shutil.copy2(src_lbl_path, dest_lbl_path)
    
    for img in img_list :
        src_img_path = os.path.join(IMG_DIR_PATH, img)
        dest_img_path = os.path.join(dest_img_folder, img)
        shutil.copy2(src_img_path, dest_img_path)"""

    
    """
    #split_yolo_dataset("../Construction-Site-Safety/data")
    proxy_lbls = get_proxy_lbl(SORTED_LBL_LIST, LBL_DIR_PATH)
    valid_images, valid_labels, valid_proxy_labels = filter_rare_classes(SORTED_IMG_LIST, 
                                                                        SORTED_LBL_LIST,
                                                                        proxy_lbls,
                                                                        IMG_DIR_PATH,
                                                                        LBL_DIR_PATH)
    
    """
    """                                                     
    state_flag = stratified_test_split(images_dir=IMG_DIR_PATH, labels_dir=LBL_DIR_PATH, image_files= valid_images, label_files=valid_labels,proxy_labels=valid_proxy_labels)
    print(state_flag)
    if state_flag:
    img_dir = "splits/kfold_base/images"
    lbl_dir = "splits/kfold_base/labels"
    sorted_img_list = sorted(os.listdir(img_dir))
    sorted_lbl_list = sorted(os.listdir(lbl_dir))
    fold_proxy_lbl = get_proxy_lbl(sorted_lbl_list=sorted_lbl_list, lbl_folder_path=lbl_dir)
    generate_folds(5,
                    sorted_img_list= sorted_img_list,
                    sorted_lbl_list=sorted_lbl_list,
                    proxy_lbls=fold_proxy_lbl,
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    output_dir="splits/kfold_base/")"""
    