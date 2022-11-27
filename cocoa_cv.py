import cv2 as cv
import numpy as np
import imutils
from matplotlib import pyplot as plt

class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    self.numPoints = numPoints
    self.radius = radius

  def describe(self, image, eps = 1e-7):
    lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints + 2))

    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)

    return hist, lbp

image_partial_ferm = cv.imread(r"cocoa_final_dataset\train\partial_fermented\partial_fermented_1.JPG")
hsv_image_pf = cv.cvtColor(image_partial_ferm, cv.COLOR_BGR2HSV)
lab_image_pf = cv.cvtColor(image_partial_ferm, cv.COLOR_BGR2LAB)

image_fully_ferm = cv.imread(r"cocoa_final_dataset\train\fully_fermented\fully_fermented_1.JPG")
hsv_image_ff = cv.cvtColor(image_fully_ferm, cv.COLOR_BGR2HSV)
lab_image_ff = cv.cvtColor(image_fully_ferm, cv.COLOR_BGR2LAB)

image_under_ferm = cv.imread(r"cocoa_final_dataset\train\under_fermented\violeta (1).JPG")
hsv_image_uf = cv.cvtColor(image_under_ferm, cv.COLOR_BGR2HSV)
lab_image_uf = cv.cvtColor(image_under_ferm, cv.COLOR_BGR2LAB)

image_no_ferm = cv.imread(r"cocoa_final_dataset\train\unfermented\ardosia_fungo (1).JPG")
hsv_image_nf = cv.cvtColor(image_no_ferm, cv.COLOR_BGR2HSV)
lab_image_nf = cv.cvtColor(image_no_ferm, cv.COLOR_BGR2LAB)

cv.namedWindow("ff", cv.WINDOW_NORMAL) 
cv.imshow("ff",hsv_image_ff)

cv.namedWindow("pf", cv.WINDOW_NORMAL) 
cv.imshow("pf",hsv_image_pf)

cv.namedWindow("uf", cv.WINDOW_NORMAL) 
cv.imshow("uf",hsv_image_uf)

cv.namedWindow("nf", cv.WINDOW_NORMAL) 
cv.imshow("nf",hsv_image_nf)
cv.waitKey(0)