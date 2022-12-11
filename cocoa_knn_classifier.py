import numpy as np 
from scipy.stats import kurtosis, skew, tvar, tmean
import cv2
import pickle

label_name = ['fullyfermented', 'partialfermented', 'underfermented', 'unfermented']

loaded_model = pickle.load(open('knnpickle_file', 'rb'))

f = []
#img = cv2.imread(r'cocoa_final_datasets\train\partial_fermented\partial_fermented_49.JPG')
#img = cv2.imread(r'cocoa_final_datasets\train\under_fermented\under_fermented_29.JPG')
img = cv2.imread(r'ROI_6.png')
#img = cv2.imread(r'cocoa_final_datasets\train\fully_fermented\fully_fermented_99.JPG')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
opening = cv2.bitwise_not(opening)
cropped_image = cv2.bitwise_and(img, img, mask = opening)
cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)

for a in range(3):
    f.append(kurtosis(cropped_image[:,:,a], axis=None))
    f.append(skew(cropped_image[:,:,a], axis=None, bias=True))
    f.append(tvar(cropped_image[:,:,a], axis=None))
    f.append(tmean(cropped_image[:,:,a], axis=None))
    print('Getting Color Values: ', a)

f_array = np.array(f, ndmin=2)
prediction = loaded_model.predict(f_array)
print(prediction)
print(label_name[int(prediction[0][0])])