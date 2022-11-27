
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# ------------------------ Data Normalization menggunakan Decimal Scaling --------------------------------
def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)



glcm_df = pd.read_csv("cocoa_features.csv")

print(glcm_df.head())

label_distr = glcm_df['label'].value_counts()

label_name = ['fullyfermented', 'partialfermented', 'underfermented', 'unfermented']

print(label_distr)

X = decimal_scaling(
            glcm_df[['dissimilarity_0', 'dissimilarity_45', 'dissimilarity_90', 'dissimilarity_135', 
                     'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135', 
                     'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135', 
                     'contrast_0', 'contrast_45', 'contrast_90', 'contrast_135', 
                     'ASM_0', 'ASM_45', 'ASM_90', 'ASM_135',
                     'energy_0', 'energy_45', 'energy_90', 'energy_135']].values
                )

print(X)