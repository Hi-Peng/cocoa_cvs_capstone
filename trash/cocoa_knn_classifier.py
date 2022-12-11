import numpy as np 
from scipy.stats import kurtosis, skew, tvar, tmean
import cv2
import pickle

label_name = ['fullyfermented', 'partialfermented', 'underfermented', 'unfermented']

loaded_model = pickle.load(open('knnpickle_file', 'rb'))

f = []
#img = cv2.imread(r'cocoa_final_dataset\train\partial_fermented\partial_fermented_49.JPG')
#img = cv2.imread(r'cocoa_final_dataset\train\under_fermented\under_fermented_29.JPG')
#img = cv2.imread(r'cocoa_final_dataset\train\under_fermented\under_fermented_1.JPG')
img = cv2.imread(r'cocoa_final_dataset\train\fully_fermented\fully_fermented_99.JPG')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
opening = cv2.bitwise_not(opening)
cropped_image = cv2.bitwise_and(img, img, mask = opening)
cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)

lab_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
hsv_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

for a in range(3):
    f.append(kurtosis(cropped_image[:,:,a], axis=None))
    f.append(skew(cropped_image[:,:,a], axis=None, bias=True))
    f.append(tvar(cropped_image[:,:,a], axis=None))
    f.append(tmean(cropped_image[:,:,a], axis=None))
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
print(prediction)

for i in range(len(f_array)):
 print("X=%s, Predicted=%s" % (f_array[i], prediction[i]))