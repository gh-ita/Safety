#Stratified k-fold splitting 
import os 
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import shutil
import numpy as np 

LBL_FOLDER_PATH = "../Construction-Site-Safety/data/labels"
SORTED_LBL_LIST = sorted(os.listdir(LBL_FOLDER_PATH))


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
        proxy_lbls[index] = obj_lst.max()
    return proxy_lbls

def stratified_test_split(
    images_dir,
    labels_dir,
    proxy_labels, 
    test_ratio=0.2,
    output_dir="splits",
    num_classes=11,
    seed=42
):
    image_files = sorted([f for f in os.listdir(images_dir)])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
    assert len(image_files) == len(label_files)

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
        
        for split in ['train', 'val']:
            os.makedirs(os.path.join(fold_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, 'labels', split), exist_ok=True)

        for idx, split in [(train_idx, 'train'), (val_idx, 'val')]:
            for img_name, lbl_name in zip(np.array(sorted_img_list)[idx], np.array(sorted_lbl_list)[idx]):
                src_img = os.path.join(images_dir, img_name)
                dst_img = os.path.join(fold_dir, 'images', split, img_name)
                shutil.copyfile(src_img, dst_img)
                src_lbl = os.path.join(labels_dir, lbl_name)
                dst_lbl = os.path.join(fold_dir, 'labels', split, lbl_name)
                shutil.copyfile(src_lbl, dst_lbl)

        print("Train:", len(train_idx), "Validation:", len(val_idx), "copied successfully.")
