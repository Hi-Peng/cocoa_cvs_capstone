import cv2
import numpy as np
from scipy.stats import kurtosis, skew, tvar, tmean

import keras
from keras import backend as K
from skimage.feature import graycomatrix, graycoprops
from datetime import datetime

properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
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
    
    return feature

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

label_name = ['fully fermented', 'partial fermented', 'under fermented', 'unfermented']

loaded_model = keras.models.load_model('cocoa_ann_trained_model.h5', custom_objects={"recall": recall, "precision": precision})

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 0.5
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 1
# Load image, grayscale, Otsu's threshold
image = cv2.imread(r'test_images\PARSIAL FERMENTED-TANPA ALAS.jpg')
image = cv2.resize(image, (0,0), fx=0.7, fy=0.7) 

height, width, channels = image.shape
print("getting image at \n", width, height)
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
cnts = cv2.findContours(opening, cv2.RETR_TREE, 2)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]

opening = cv2.bitwise_not(opening)
original = cv2.bitwise_and(original, original, mask=opening)
cv2.waitKey()

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)

    if h > 1000 or w > 1000:
        print("Big picture passed")
        

    elif h > 50 and w > 50:
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI = original[y:y+h, x:x+w]

        lab_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB)
        hsv_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        glcm_f = []
        f = []

        for a in range(3):
            f.append(kurtosis(ROI[:,:,a], axis=None))
            f.append(skew(ROI[:,:,a], axis=None, bias=True))
            f.append(tvar(ROI[:,:,a], axis=None))
            f.append(tmean(ROI[:,:,a], axis=None))

            
            f.append(kurtosis(hsv_img[:,:,a], axis=None))
            f.append(skew(hsv_img[:,:,a], axis=None, bias=True))
            f.append(tvar(hsv_img[:,:,a], axis=None))
            f.append(tmean(hsv_img[:,:,a], axis=None))

            f.append(kurtosis(lab_img[:,:,a], axis=None))
            f.append(skew(lab_img[:,:,a], axis=None, bias=True))
            f.append(tvar(lab_img[:,:,a], axis=None))
            f.append(tmean(lab_img[:,:,a], axis=None))
            
        glcm_f = calc_glcm_all_agls(gray_img, props=properties)
        for u in range(len(glcm_f)):
            f.append(glcm_f[u])

        f_array = np.array(f, ndmin=2)
        prediction = loaded_model.predict(f_array)

        ROI = cv2.putText(image, str(label_name[int(prediction[0][0])]),  (x, y), font, 
                    fontScale, color, thickness, cv2.LINE_AA)

        # cv2.imwrite('result/ROI_{}_{}.png'.format(str(label_name[int(prediction[0][0])]), ROI_number), ROI)

        print("detected " + str(label_name[int(prediction[0][0])]) + " cocoa")
        ROI_number += 1
    else:
        pass

now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")
print(dt_string)
cv2.imwrite('result/FINAL-{}.png'.format(str(dt_string)), image)
cv2.imshow("classified", image)
cv2.waitKey(0)