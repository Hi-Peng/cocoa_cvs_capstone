
# import packages
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import cv2
import os

def print_progress(val, val_len, folder, sub_folder, filename, bar_size=10):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, filename), end="\r")

# reading main image
img1 = cv2.imread(r"cocoa_final_dataset\train\fully_fermented\fully_fermented_1.JPG")
  
# checking the number of channels
print('No of Channel is: ' + str(img1.ndim))
  
# reading reference image
img2 = cv2.imread(r"result\ROI_fully fermented_2.png")
  
# checking the number of channels
print('No of Channel is: ' + str(img2.ndim))
  
image = img1
reference = img2
  
dataset_dir = 'cocoa_final_dataset\\'
for folder in os.listdir(dataset_dir):
    for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        sub_folder = 'unfermented'
        sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))
        len_sub_folder = len(sub_folder_files) - 1
        print(sub_folder)
        for i, filename in enumerate(sub_folder_files):
            # Read images
            print(os.path.join(dataset_dir, folder, sub_folder, filename))
            img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))
            matched = match_histograms(img, reference,
                           multichannel=True)
            cv2.imwrite('cocoa_final_dataset_1/train/{}/{}'.format(sub_folder, filename), matched)
            print_progress(i, len_sub_folder, folder, sub_folder, filename)