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

IMAGE_FOLDER = "Construction-Site-Safety/data/images"
LABEL_FOLDER = "Construction-Site-Safety/data/labels"
IMG_FILE_LIST = sorted(os.listdir(IMAGE_FOLDER))
LBL_FILE_LIST = sorted(os.listdir(LABEL_FOLDER))
print(len(IMG_FILE_LIST))

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
    
def remove_data(file_name_list, img_list, label_list, img_folder, label_folder) :
    """
    Removes the images and labels of the data that doesn't contain any annotations
    file_name_list : list of the files to erase
    img_list : full list of images
    label_list : full list of labels
    """
    for elem in file_name_list.values():
        img_name = img_list[elem]
        label_name = label_list[elem]
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

redundancy_list = find_redundant_images(IMG_FILE_LIST, IMAGE_FOLDER)
count = 0
for img_list in redundancy_list.values():
    if len(img_list) > 1 :
        count += 1
print(count)

"""check_image_label_index_similarity(LABEL_FOLDER, IMAGE_FOLDER)
empty_labels = empty_label_finder(LABEL_FOLDER)
remove_data(empty_labels, IMG_FILE_LIST, LBL_FILE_LIST, IMAGE_FOLDER, LABEL_FOLDER)

grid_size = (20,10)
fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(40, 40))
axes = axes.flatten()
count = 0
for elem in empty_labels.values():  
    count += 1
    image = Image.open(os.path.join(IMAGE_FOLDER, img_file_list[elem]))
    axes[count].axis('off')
    axes[count].imshow(image)
    
plt.tight_layout()
plt.show()"""
