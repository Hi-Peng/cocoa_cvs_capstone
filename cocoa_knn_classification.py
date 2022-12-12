import cv2
import numpy as np
from scipy.stats import kurtosis, skew, tvar, tmean
import pickle
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from datetime import datetime

properties = ['energy']

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

def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)

label_name = ['fully fermented', 'partial fermented', 'under fermented', 'unfermented']

col = [ 
        'l_kurtosis', 'l_skew', 'l_tvar', 'l_tmean','l_entropy',
        'a_kurtosis', 'a_skew', 'a_tvar', 'a_tmean','a_entropy',
        'b_kurtosis', 'b_skew', 'b_tvar', 'b_tmean','b_entropy',
        
        'energy_0',         'energy_45',        'energy_90',        'energy_135', 

        'label'
]

loaded_model = pickle.load(open('cocoa_knn_trained_model_1', 'rb'))

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
image = cv2.imread(r'test_images\NON FERMENTED-TANPA ALAS.jpg')
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 

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

predicts = []
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)

    if h > 1000 or w > 1000:
        print("big picture skipped")

    elif h > 50 and w > 50:
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI = original[y:y+h, x:x+w]

        lab_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB)
        hsv_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        f = []

        for a in range(3):
            f.append(kurtosis(lab_img[:,:,a], axis=None))
            f.append(skew(lab_img[:,:,a], axis=None, bias=True))
            f.append(tvar(lab_img[:,:,a], axis=None))
            f.append(tmean(lab_img[:,:,a], axis=None))
            f.append(shannon_entropy(lab_img[:,:,a])) 

        glcm_f = calc_glcm_all_agls(gray_img, props=properties)
        for u in range(len(glcm_f)):
            f.append(glcm_f[u])
        predicts.append(f)
        f_array = np.array(f, ndmin=2)
        ROI_number += 1
    else:
        pass

prediction = loaded_model.predict(predicts)
prediction_label_index = [np.where(r==1)[0][0] for r in prediction]
print(prediction)

ROI_number = 0

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)

    if h > 1000 or w > 1000:
        print("big picture skipped")

    elif h > 50 and w > 50:
        ROI = original[y:y+h, x:x+w]
        idx = prediction_label_index[ROI_number]
        ROI = cv2.putText(image, str(label_name[idx]),  (x, y), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        ROI_number += 1
    else:
        pass

now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")
print(dt_string)

cv2.imwrite('result/FINAL-{}.png'.format(str(dt_string)), image)
cv2.imshow("classified", image)
cv2.waitKey(0)