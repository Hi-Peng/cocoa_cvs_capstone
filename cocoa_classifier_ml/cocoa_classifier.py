# Importing required modules

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation

import keras
from keras import backend as K

import itertools

import cv2

from scipy.stats import kurtosis, skew, tvar, tmax, tmin, tmean
class cocoa_image_segmentation:
    def __init__(self, image)

def show_result(winname, img, wait_time):
    scale = 0.2
    disp_img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(winname, disp_img)
    cv2.waitKey(wait_time)

img = cv2.imread('CB_001.JPG')
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
u_green = np.array([58-50, 185-50, 106-50])
l_green = np.array([58+50, 185+50, 106+50])

# Threshold the HSV image to extract green color
mask = cv2.inRange(img, l_green, u_green)
# mask = cv2.bitwise_not(mask)

#cv2.imwrite('mask.png', mask)
show_result('mask', img_lab, 0)
cv2.destroyAllWindows()