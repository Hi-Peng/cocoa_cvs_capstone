
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential, save_model
from keras.layers import Dense, Activation

import keras
from keras import backend as K

import itertools
import pickle

# ------------------------ Data Normalization menggunakan Decimal Scaling --------------------------------
def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)

# --------------------- create custom metric evaluation ---------------------
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_model_(history):
    names = [['accuracy', 'val_accuracy'], 
             ['loss', 'val_loss'], 
             ['precision', 'val_precision'], 
             ['recall', 'val_recall']]
    for name in names :
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()

# Read extracted features
glcm_df = pd.read_csv("cocoa_features_lab.csv")

print(glcm_df.head())

label_distr = glcm_df['label'].value_counts()

label_name = ['fullyfermented', 'partialfermented', 'underfermented', 'unfermented']

print(label_distr)

X = decimal_scaling(
            glcm_df[[
                        'l_kurtosis', 'l_skew', 'l_tvar', 'l_tmean','l_entropy',
                        'a_kurtosis', 'a_skew', 'a_tvar', 'a_tmean','a_entropy',
                        'b_kurtosis', 'b_skew', 'b_tvar', 'b_tmean','b_entropy',

                        'dissimilarity_0',  'dissimilarity_45', 'dissimilarity_90', 'dissimilarity_135',
                        'correlation_0',    'correlation_45',   'correlation_90',   'correlation_135',
                        'homogeneity_0',    'homogeneity_45',   'homogeneity_90',   'homogeneity_135',
                        'contrast_0',       'contrast_45',      'contrast_90',      'contrast_135',
                        'ASM_0',            'ASM_45',           'ASM_90',           'ASM_135',
                        'energy_0',         'energy_45',         'energy_90',         'energy_135']].values
)

le = LabelEncoder()
le.fit(glcm_df["label"].values)

print(" categorical label : \n", le.classes_)

Y = le.transform(glcm_df['label'].values)
Y = to_categorical(Y)

print("\n\n one hot encoding for sample 0 : \n", Y)

X_train, X_test, y_train, y_test = \
                    train_test_split(X, 
                                     Y, 
                                     test_size=0.25, 
                                     random_state=42)
  
print("Dimensi data :\n")
print("X train \t X test \t Y train \t Y test")  
print("%s \t %s \t %s \t %s" % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

max_len = X_train.shape[1]  

knn = KNeighborsClassifier(3)
# model = nn_model(max_len)
knn.fit(X_train, y_train)

knnPickle = open('cocoa_knn_trained_model_1', 'wb') 
pickle.dump(knn, knnPickle)

knnPickle.close()

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))