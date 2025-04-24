#mask, gloves, no glove, goggles, no goggles 
import os 
import shutil

LBL_FOLDER = "../data/augmentation data/labels"

def extract_imgs(clss_list, lbl_folder):
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

    
if __name__ == "__main__" :
    merge_data(None, src_img_folder="../data/data/augmented_negatives/images",
               src_lbl_folder="../data/data/augmented_negatives/labels",
               trgt_img_folder="../data/data/images",
               trgt_lbl_folder="../data/data/labels")

    