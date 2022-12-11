import numpy as np
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from skimage.io import imread, imshow
from scipy.stats import kurtosis, skew, tvar, tmax, tmin, tmean
from PIL import Image, ImageStat
import matplotlib.pyplot as plt

def print_image_summary(image, labels):
    
    print('--------------')
    print('Image Details:')
    print('--------------')
    print(f'Image dimensions: {image.shape}')
    print('Channels:')
    
    if len(labels) == 1:
        image = image[..., np.newaxis]
        
    for i, lab in enumerate(labels):
        min_val = np.min(image[:,:,i])
        max_val = np.max(image[:,:,i])
        print(f'{lab} : min={min_val:.4f}, max={max_val:.4f}')
        print(kurtosis(image[:,:,i], axis=None, bias=True))
        print(skew(image[:,:,i], axis=None, bias=True))
        print(tvar(image[:,:,i], axis=None))
        print(tmean(image[:,:,i], axis=None))
        print(tmin(image[:,:,i], axis=None))


image_rgb = imread('fully_fermented_1.JPG')
image_lab = rgb2lab(image_rgb / 255)
print_image_summary(image_lab, ['R', 'G', 'B'])