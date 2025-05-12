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
#IMG_FILE_LIST = sorted(os.listdir(IMAGE_FOLDER))
#LBL_FILE_LIST = sorted(os.listdir(LABEL_FOLDER))

p_or_lbl_map = {0:"gloves", 1:"goggles", 2:"helmet", 3:"mask", 9:"person", 10:"safety-vest"}
n_or_lbl_map = {4:"no-gloves", 5: "no-goggles", 6 : "no-helmet", 7:"no-mask", 8:"no-safety-vest", 9:"person"}

p_lbl_map =  {"gloves":0, "goggles":1,"helmet" :2,"mask":3,"person":4, "safety-vest":5}
n_lbl_map =  {"no-gloves":0, "no-goggles":1, "no-helmet":2, "no-mask":3, "no-safety-vest":4, "person":5}

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
    print(f"removed {count} images")
    
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


def remove_files_from_folder(folder_path, files_to_remove):
    """
    Removes specified files from a given folder.

    Parameters:
    - folder_path (str): Path to the folder containing the files.
    - files_to_remove (list): List of filenames (not full paths) to remove.

    Returns:
    - removed_files (list): List of successfully removed files.
    - not_found_files (list): List of files that were not found.
    """
    removed_files = []
    not_found_files = []

    for filename in files_to_remove:
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            removed_files.append(filename)
        else:
            not_found_files.append(filename)
    print(len(removed_files))
    return removed_files, not_found_files


def fix_class_ids_in_labels(labels_dir):
    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue  

        label_path = os.path.join(labels_dir, filename)
        new_lines = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip malformed lines


                try:
                    class_id = str(int(float(parts[0])))  # safe conversion
                    rest = parts[1:]
                    new_line = " ".join([class_id] + rest)
                    new_lines.append(new_line)
                except ValueError:
                    print(f"Warning: Skipping invalid line in {filename}: {line.strip()}")

        # Overwrite file with corrected lines
        with open(label_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

    print("✅ Class ID corrections complete.")



                

if __name__ == "__main__":
    #n
    lbl_folder_path = "../data/merged_data/augmented_data_2/n_clss/labels"
    lbl_list = os.listdir(lbl_folder_path)
    remap_data(or_lbl_map = n_or_lbl_map, trgt_lbl_map = n_lbl_map, 
               lbl_folder = lbl_folder_path, lbl_list = lbl_list)