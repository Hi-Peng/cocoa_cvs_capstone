import cv2
import numpy as np
from scipy.stats import kurtosis, skew, tvar, tmean
import pickle

label_name = ['fullyfermented', 'partialfermented', 'underfermented', 'unfermented']

loaded_model = pickle.load(open('knnpickle_file', 'rb'))

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
image = cv2.imread(r'FULLY FERMENTED-TANPA ALAS.jpg')
image = cv2.resize(image, (0,0), fx=0.2, fy=0.2) 
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
cv2.imshow('image', original)
cv2.waitKey()

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
    ROI = original[y:y+h, x:x+w]

    lab_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB)
    hsv_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
    f = []
    for a in range(3):
        f.append(kurtosis(ROI[:,:,a], axis=None))
        f.append(skew(ROI[:,:,a], axis=None, bias=True))
        f.append(tvar(ROI[:,:,a], axis=None))
        f.append(tmean(ROI[:,:,a], axis=None))
        f.append(kurtosis(lab_img[:,:,a], axis=None))
        f.append(skew(lab_img[:,:,a], axis=None, bias=True))
        f.append(tvar(lab_img[:,:,a], axis=None))
        f.append(tmean(lab_img[:,:,a], axis=None))
        f.append(kurtosis(hsv_img[:,:,a], axis=None))
        f.append(skew(hsv_img[:,:,a], axis=None, bias=True))
        f.append(tvar(hsv_img[:,:,a], axis=None))
        f.append(tmean(hsv_img[:,:,a], axis=None))

    f_array = np.array(f, ndmin=2)
    prediction = loaded_model.predict(f_array)

    print(x)

    ROI = cv2.putText(ROI, str(prediction),  (5, 10), font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)

    ROI_number += 1
    