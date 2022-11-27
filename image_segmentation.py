import cv2 as cv
import numpy as np
import imutils
from matplotlib import pyplot as plt

is_normalized = True
image = cv.imread("IMG_1483.JPG")
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Set total number of bins in the histogram
bins_num = 256
 
# Get the image histogram
hist, bin_edges = np.histogram(image, bins=bins_num)
 
# Get normalized histogram if it is required
if is_normalized:
    hist = np.divide(hist.ravel(), hist.max())
 
# Calculate centers of bins
bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
 
# Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
weight1 = np.cumsum(hist)
weight2 = np.cumsum(hist[::-1])[::-1]
 
# Get the class means mu0(t)
mean1 = np.cumsum(hist * bin_mids) / weight1
# Get the class means mu1(t)
mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
 
inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
 
# Maximize the inter_class_variance function val
index_of_max_val = np.argmax(inter_class_variance)
 
threshold = bin_mids[:-1][index_of_max_val]
print("Otsu's algorithm implementation thresholding result: ", threshold)

otsu_threshold, image_result = cv.threshold(
    image, threshold, 255, cv.THRESH_OTSU
)
print("Obtained threshold: ", otsu_threshold)
mask_inv = image_result
mask_inv[mask_inv==255] = 10
mask_inv[mask_inv==0] = 255
mask_inv[mask_inv==10] = 0
output = cv.bitwise_and(image,image, mask= mask_inv)
cv.namedWindow("imageout", cv.WINDOW_NORMAL) 
cv.imshow("imageout",hsv_image)
cv.waitKey(0)