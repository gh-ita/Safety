#remove empty images 
#empty image = image with empty label files 
"""pipeline :
1- sort the images and labels folders 
2- store the empty label files names along with their index 
3- erase the images and labels 
"""
"""
Find redundant data :
analyse the images arrays 
"""
import os
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
import hashlib
from ultralytics.data.utils import visualize_image_annotations

IMAGE_FOLDER = "../data/augmentation data/images"
LABEL_FOLDER = "../data/augmentation data/labels"
IMG_FILE_LIST = sorted(os.listdir(IMAGE_FOLDER))
LBL_FILE_LIST = sorted(os.listdir(LABEL_FOLDER))
label_map =  {1:"gloves", 2:"goggles",3:"helmet",4:"mask",5:"no-gloves", 6:"no-goggles",7:"no-helmet",8:"no-mask",9:"no-safety-vest",10:"person", 11:"safety-vest"}
trgt_lbl_map = {"gloves":0, "goggles" : 1, "helmet" :2, "mask":3, "no-gloves" :4, "no-goggles": 5, "no-helmet":6, "no-mask":7, "no-safety-vest" :8, "person" : 9, "safety-vest" : 10}
def empty_label_finder(label_folder_path):
    """
    The label folder and image folder must contain the same number of corresponding images and their label files
    Each image and its label file must have the same name
    empty_labels dictionnary contains the empty files names, along with their index in the folder
    """
    label_list = sorted(os.listdir(label_folder_path))
    empty_labels = {os.path.splitext(file)[0] : index for index, file in enumerate(label_list) if os.path.getsize(os.path.join(label_folder_path,file)) == 0}
    return empty_labels 

def check_image_label_index_similarity(label_folder_path, image_folder_path):
    """
    Quality checking method, checks whether the images and 
    their corresponding labels files have the same index in their respective folders
    """
    image_file_name_list = sorted(os.listdir(label_folder_path))
    label_file_name_list = sorted(os.listdir(image_folder_path))
    same_order = True
    for image_name, label_name in zip(image_file_name_list, label_file_name_list) :
        if os.path.splitext(image_name)[0] != os.path.splitext(label_name)[0] :
            same_order = False
    print(same_order)
    
def remove_bcgd_data(file_name_list, img_list, label_list, img_folder, label_folder) :
    """
    Removes the images and labels of the data that doesn't contain any annotations
    file_name_list : list of the files to erase
    img_list : full list of images
    label_list : full list of labels
    """
    for elem in file_name_list.values():
        if isinstance(elem, int):
            img_name = img_list[elem]
            label_name = label_list[elem]
            os.remove(os.path.join(img_folder, img_name))
            os.remove(os.path.join(label_folder, label_name))
        elif isinstance(elem, tuple):
            img_name = img_list[elem[1]]
            label_name = label_list[elem[1]]
            os.remove(os.path.join(img_folder, img_name))
            os.remove(os.path.join(label_folder, label_name))
    print(f"Removed {len(file_name_list)} files, new image folder size {len(os.listdir(img_folder))}, new label folder size {len(os.listdir(label_folder))}")

def find_redundant_images(img_list, img_folder):
    img_instances = {}
    for img_filename in img_list:
        img = Image.open(os.path.join(img_folder, img_filename))
        hist = np.array(img.convert('L').histogram())
        hist_tuple = tuple(hist) 
        img_hash = hashlib.md5(np.array(hist_tuple).tobytes()).hexdigest()
        if img_hash not in img_instances :
            img_instances[img_hash] = [img_filename]
        else :
            img_instances[img_hash].append(img_filename)
    return img_instances

def get_img_class_combs(class_comb_list, img_list, lbl_folder, lbl_list):
    """
    returns the images per combination of objects
    """
    class_comb_img = {}
    for index, (img, lbl) in enumerate(zip(img_list, lbl_list)) :
        with open(os.path.join(lbl_folder, lbl)) as lbl_file :
            img_objs = []
            for line in lbl_file:
                img_objs.append(int(line.split()[0]))
        class_set = frozenset(img_objs)
        if class_set in class_comb_img :
            class_comb_img[class_set].append((img,index))
        else :
            class_comb_img[class_set] = [(img,index)]
    if frozenset(class_comb_list) in class_comb_img :
        return class_comb_img[frozenset(class_comb_list)]
    else :
        print("Class combination already erased")
        return None

def remove_clss(clss_id, img_folder,lbl_folder, img_list):
    count = 0
    files_to_remove = []
    for img in img_list :
        lbl_filename = os.path.splitext(img)[0]+ ".txt"
        with open(os.path.join(lbl_folder, lbl_filename)) as lbl_file :
            for line in lbl_file :
                if int(line.split()[0]) == clss_id :
                    files_to_remove.append(img)
                    break
    for file in files_to_remove :
        count += 1
        print(count)
        lbl_filename = os.path.splitext(file)[0]+ ".txt"
        os.remove(os.path.join(img_folder, file))
        os.remove(os.path.join(lbl_folder, lbl_filename))
    
def remap_data(or_lbl_map, trgt_lbl_map, lbl_folder, lbl_list):
    count = 0
    for lbl_file in lbl_list :
        new_lines = []
        with open(os.path.join(lbl_folder, lbl_file), "r") as file :
            for line in file :
                parts = line.strip().split()
                or_clss = int(parts[0])
                trgt_clss = trgt_lbl_map[or_lbl_map[or_clss]]
                new_line = " ".join([str(trgt_clss)]+parts[1:])
                new_lines.append(new_line)
        with open(os.path.join(lbl_folder, lbl_file), "w") as file:
            for line in new_lines:
                file.write(line + "\n")
        count += 1
    print(f"remaped {count} files")
                
                

if __name__ == "__main__":
    #remove_clss(0,IMAGE_FOLDER, LABEL_FOLDER, IMG_FILE_LIST)
    remap_data(or_lbl_map=label_map, trgt_lbl_map=trgt_lbl_map, lbl_folder=LABEL_FOLDER, lbl_list=LBL_FILE_LIST)