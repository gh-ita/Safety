#mask, gloves, no glove, goggles, no goggles 
import os 
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


LBL_FOLDER = "../data/augmentation data/labels"

def extract_imgs(clss_list, lbl_folder):
    """
    returns the filenames
    """
    file_list = set()
    lbl_list = os.listdir(lbl_folder)
    for clss_id in clss_list :
        for lbl_file in lbl_list :
            with open(os.path.join(lbl_folder, lbl_file), "r") as file :
                for line in file :
                    if int(line.split()[0]) == clss_id :
                        file_list.add(os.path.splitext(lbl_file)[0])
    print(len(file_list))
    return file_list



def merge_data(img_list, trgt_img_folder, trgt_lbl_folder,
               src_img_folder=None, src_lbl_folder=None,
               img_ext=".jpg"):
    """
    Copy images and labels to their target folders using base names.

    Args:
        img_list (list): List of base names (no extensions).
        trgt_img_folder (str): Path to target image folder.
        trgt_lbl_folder (str): Path to target label folder.
        src_img_folder (str): Path to source image folder (optional).
        src_lbl_folder (str): Path to source label folder (optional).
        img_ext (str): Image file extension (default: .jpg)
    """
    count = 0
    if not img_list:
        for filename in os.listdir(src_img_folder):
            src_img_path = os.path.join(src_img_folder, filename)
            dst_img_path = os.path.join(trgt_img_folder, filename)
            
            if os.path.isfile(src_img_path): 
                shutil.copy2(src_img_path, dst_img_path)
                print(f"Copied: {src_img_path} → {dst_img_path}")
        for filename in os.listdir(src_lbl_folder):
            src_lbl_path = os.path.join(src_lbl_folder, filename)
            dst_lbl_path = os.path.join(trgt_lbl_folder, filename)
            
            if os.path.isfile(src_lbl_path): 
                shutil.copy2(src_lbl_path, dst_lbl_path)
                print(f"Copied: {src_lbl_path} → {dst_lbl_path}")
    else :       
        for base_name in img_list:
            count += 1
            img_file = base_name + img_ext
            lbl_file = base_name + ".txt"

            # Full source paths
            src_img_path = os.path.join(src_img_folder, img_file) if src_img_folder else img_file
            src_lbl_path = os.path.join(src_lbl_folder, lbl_file) if src_lbl_folder else lbl_file

            # Full destination paths
            dst_img_path = os.path.join(trgt_img_folder, img_file)
            dst_lbl_path = os.path.join(trgt_lbl_folder, lbl_file)

            # Copy image
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Image not found: {src_img_path}")

            # Copy label
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, dst_lbl_path)
            else:
                print(f"Label not found: {src_lbl_path}")
    print(count)



def copy_files(source_folder, destination_folder, files_to_copy=None):
    """
    Copies files from the source folder to the destination folder.

    Parameters:
    - source_folder (str): Path to the folder containing the files.
    - destination_folder (str): Path to the folder where files will be copied.
    - files_to_copy (list, optional): List of filenames to copy. 
      If None, all files in the source folder will be copied.

    Returns:
    - copied_files (list): List of files successfully copied.
    - not_found_files (list): List of files not found in the source folder.
    """
    copied_files = []
    not_found_files = []

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    if files_to_copy is None:
        files_to_copy = [
            f for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f))
        ]

    for filename in files_to_copy:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
            copied_files.append(filename)
        else:
            not_found_files.append(filename)
    print(len(copied_files))
    return copied_files, not_found_files


def show_images_one_by_one(filenames, folder_path, delay=0):
    for filename in filenames:
        full_path = os.path.join(folder_path, filename+".jpg")
        img = mpimg.imread(full_path)

        plt.imshow(img)
        plt.axis('off')
        plt.title(filename)
        plt.show()
        
        time.sleep(delay)  
        plt.close()


    
if __name__ == "__main__" :
    #copy_files("PPE_Detection/data/yolo/val/images","PPE_Detection/data/images")
    
    merge_data(None, src_img_folder="../data/merged_data/augmented_data_2/no-goggles/images",
               src_lbl_folder="../data/merged_data/augmented_data_2/no-goggles/labels",
               trgt_img_folder="../data/merged_data/augmented_data_2/images",
               trgt_lbl_folder="../data/merged_data/augmented_data_2/labels")


    