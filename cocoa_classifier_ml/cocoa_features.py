import pandas as pd 
import numpy as np 
from scipy.stats import kurtosis, skew, tvar, tmax, tmin, tmean

import matplotlib.pyplot as plt
import cv2
import os
import re

class cocoa_features:
    def __init__(self, dataset_dir):
        self.features = []
        self.labels = []
        self.descs = []
        self.dataset_dir = None
        self.features_sets = None

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
    
    def extract_features(dataset_dir):
        for folder in os.listdir(dataset_dir):
            for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
                sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))
                len_sub_folder = len(sub_folder_files) - 1
                for i, filename in enumerate(sub_folder_files):
                    f = []
                    img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))

                    cropped_image = img

                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)

                    for i in range(3):
                        f.append(kurtosis(cropped_image[:,:,i], axis=None))
                        f.append(skew(cropped_image[:,:,i], axis=None, bias=True))
                        f.append(tvar(cropped_image[:,:,i], axis=None))
                        f.append(tmean(cropped_image[:,:,i], axis=None))
                    f.append(normalize_label(os.path.splitext(filename)[0]))

                    self.features.append(f)
                    # print(f)
                    self.labels.append(normalize_label(os.path.splitext(filename)[0]))
                    self.descs.append(normalize_desc(folder, sub_folder))
            
                    print_progress(i, len_sub_folder, folder, sub_folder, filename)

        col = ['l_kurtosis', 'l_skew', 'l_tvar', 'l_tmean', 'a_kurtosis', 'a_skew', 'a_tvar', 'a_tmean', 'b_kurtosis', 'b_skew', 'b_tvar', 'b_tmean', 'label'] 
        
        return pd.DataFrame(features, columns = col)


