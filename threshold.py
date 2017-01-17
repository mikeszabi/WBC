# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:17:53 2017

@author: SzMike
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from defPaths import *
 
image_file=os.path.join(image_dir,'36.bmp')

# Read imageű
im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
im  = cv2.medianBlur(im,5)

ret,th1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [im, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

