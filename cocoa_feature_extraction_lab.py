import pandas as pd 
import numpy as np 
from scipy.stats import kurtosis, skew, tvar, tmax, tmin, tmean
from skimage.feature import graycomatrix, graycoprops

import matplotlib.pyplot as plt
import cv2
import os
import re

# -------------------- Utility function ------------------------
def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("","", "()"))
    str_ = str_.split("_")
    return ''.join(str_[:2])

def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder 
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text

def print_progress(val, val_len, folder, sub_folder, filename, bar_size=10):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] folder : %s/%s/ ----> file : %s" % (progr, folder, sub_folder, filename), end="\r")

# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = graycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]

    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature

dataset_dir = 'cocoa_final_dataset_1\\'

properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

features = [] #list image matrix 
labels = []
descs = []
for folder in os.listdir(dataset_dir):
    for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))
        len_sub_folder = len(sub_folder_files) - 1
        for i, filename in enumerate(sub_folder_files):
            f = []
            glcm_f = []
            # Read images
            img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))
            # Scale down images
            img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
            
            cropped_image = img
            # cv2.imshow("open",cropped_image)
            lab_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
            hsv_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
            gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            for a in range(3):
                f.append(kurtosis(cropped_image[:,:,a], axis=None))
                f.append(skew(cropped_image[:,:,a], axis=None, bias=True))
                f.append(tvar(cropped_image[:,:,a], axis=None))
                f.append(tmean(cropped_image[:,:,a], axis=None))

                f.append(kurtosis(hsv_img[:,:,a], axis=None))
                f.append(skew(hsv_img[:,:,a], axis=None, bias=True))
                f.append(tvar(hsv_img[:,:,a], axis=None))
                f.append(tmean(hsv_img[:,:,a], axis=None))

                f.append(kurtosis(lab_img[:,:,a], axis=None))
                f.append(skew(lab_img[:,:,a], axis=None, bias=True))
                f.append(tvar(lab_img[:,:,a], axis=None))
                f.append(tmean(lab_img[:,:,a], axis=None))

            #glcm_f = calc_glcm_all_agls(gray_img, normalize_label(os.path.splitext(filename)[0]), props=properties)
            #for x in range(len(glcm_f)):
            #    f.append(glcm_f[x])

            features.append(f)

            labels.append(normalize_label(os.path.splitext(filename)[0]))
            descs.append(normalize_desc(folder, sub_folder))
            print_progress(i, len_sub_folder, folder, sub_folder, filename)

col = [ 
        'b_kurtosis', 'b_skew', 'b_tvar', 'b_tmean',
        'g_kurtosis', 'g_skew', 'g_tvar', 'g_tmean',
        'r_kurtosis', 'r_skew', 'r_tvar', 'r_tmean', 
        
        'h_kurtosis', 'v_skew', 'h_tvar', 'h_tmean',
        's_kurtosis', 'v_skew', 's_tvar', 's_tmean',
        'v_kurtosis', 'v_skew', 'v_tvar', 'v_tmean', 

        'l_kurtosis', 'l_skew', 'l_tvar', 'l_tmean',
        'a_kurtosis', 'a_skew', 'a_tvar', 'a_tmean',
        'b_kurtosis', 'b_skew', 'b_tvar', 'b_tmean',

        'label'
] 

'''col = [ 
        'b_kurtosis', 'b_skew', 'b_tvar', 'b_tmean',
        'g_kurtosis', 'g_skew', 'g_tvar', 'g_tmean',
        'r_kurtosis', 'r_skew', 'r_tvar', 'r_tmean', 
        
        'h_kurtosis', 'v_skew', 'h_tvar', 'h_tmean',
        's_kurtosis', 'v_skew', 's_tvar', 's_tmean',
        'v_kurtosis', 'v_skew', 'v_tvar', 'v_tmean', 

        'l_kurtosis', 'l_skew', 'l_tvar', 'l_tmean',
        'a_kurtosis', 'a_skew', 'a_tvar', 'a_tmean',
        'b_kurtosis', 'b_skew', 'b_tvar', 'b_tmean',

        'dissimilarity_0',  'dissimilarity_45', 'dissimilarity_90', 'dissimilarity_135',
        'correlation_0',    'correlation_45',   'correlation_90',   'correlation_135',
        'homogeneity_0',    'homogeneity_45',   'homogeneity_90',   'homogeneity_135',
        'contrast_0',       'contrast_45',      'contrast_90',      'contrast_135',
        'ASM_0',            'ASM_45',           'ASM_90',           'ASM_135',
        'energy_0',         'energy_45',         'energy_90',         'energy_135',

        'label'
]'''


glcm_df = pd.DataFrame(features, 
                      columns = col)

#save to csv
glcm_df.to_csv("cocoa_features_lab.csv")

glcm_df.head(7)
