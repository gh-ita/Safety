#Stratified k-fold splitting 
import os 
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import shutil
import numpy as np 
from collections import Counter 

LBL_DIR_PATH = "Construction-Site-Safety/data/labels"
SORTED_LBL_LIST = sorted(os.listdir(LBL_DIR_PATH))
IMG_DIR_PATH = "Construction-Site-Safety/data/images"
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
    
    return valid_images, valid_labels, valid_proxy_labels

def stratified_test_split(
    images_dir,
    labels_dir,
    image_files,
    label_files,
    proxy_labels, 
    test_ratio=0.2,
    output_dir="splits",
    seed=42
):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    indices = list(range(len(image_files)))
    for train_idx, test_idx in sss.split(indices, proxy_labels):
        train_imgs = [image_files[i] for i in train_idx]
        train_lbls = [label_files[i] for i in train_idx]
        test_imgs = [image_files[i] for i in test_idx]
        test_lbls = [label_files[i] for i in test_idx]

    test_img_dir = os.path.join(output_dir, "test/images")
    test_lbl_dir = os.path.join(output_dir, "test/labels")
    base_img_dir = os.path.join(output_dir, "kfold_base/images")
    base_lbl_dir = os.path.join(output_dir, "kfold_base/labels")
    for d in [test_img_dir, test_lbl_dir, base_img_dir, base_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    for img, lbl in zip(test_imgs, test_lbls):
        shutil.copy(os.path.join(images_dir, img), os.path.join(test_img_dir, img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(test_lbl_dir, lbl))
    
    for img, lbl in zip(train_imgs, train_lbls):
        shutil.copy(os.path.join(images_dir, img), os.path.join(base_img_dir, img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(base_lbl_dir, lbl))

    print(f"Test split complete: {len(test_imgs)} test samples, {len(train_imgs)} for k-fold.")
    return True
    
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


if __name__ == "__main__":
    """
    proxy_lbls = get_proxy_lbl(SORTED_LBL_LIST, LBL_DIR_PATH)
    valid_images, valid_labels, valid_proxy_labels = filter_rare_classes(SORTED_IMG_LIST, 
                                                                        SORTED_LBL_LIST,
                                                                        proxy_lbls,
                                                                        IMG_DIR_PATH,
                                                                        LBL_DIR_PATH
                                                                        )
    state_flag = stratified_test_split(images_dir=IMG_DIR_PATH, labels_dir=LBL_DIR_PATH, image_files= valid_images, label_files=valid_labels,proxy_labels=valid_proxy_labels)
    print(state_flag)
    if state_flag:"""
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
                    output_dir="splits/kfold_base/")
