import pandas as pd 
import numpy as np 
from scipy.stats import kurtosis, skew, tvar, tmax, tmin, tmean

import matplotlib.pyplot as plt
import cv2
import os

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras import backend as K

import itertools

label_name = ['fullyfermented', 'partialfermented', 'underfermented', 'unfermented']
model = keras.models.load_model('cocoa_1.h5')

f = []
img = cv2.imread('ROI_8.png')

# cropped_image = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
cropped_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

for a in range(3):
    f.append(kurtosis(cropped_image[:,:,a], axis=None))
    f.append(skew(cropped_image[:,:,a], axis=None, bias=True))
    f.append(tvar(cropped_image[:,:,a], axis=None))
    f.append(tmean(cropped_image[:,:,a], axis=None))
    print('Getting Color Values: ', a)

print(type(np.array(f)))
f_array = np.array(f, ndmin=2)
print(f_array)

prediction = model.predict(np.array(f_array))
print(label_name[int(prediction[0][0])])