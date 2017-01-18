# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:04:41 2017

@author: SzMike
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


#get rid of very bright and very dark regions
delta=30
lower_gray = np.array([delta, delta,delta])
upper_gray = np.array([255-delta,255-delta,255-delta])
# Threshold the image to get only selected
mask = cv2.inRange(im, lower_gray, upper_gray)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(im,im, mask= mask)

cv2.imshow('res',res)
cv2.waitKey()

#Convert to HSV space
HSV_img = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
hue = HSV_img[:, :, 0]

#select maximum value of H component from histogram
hist = cv2.calcHist([hue],[0],None,[256],[0,256])
hist= hist[1:, :] #suppress black value
elem = np.argmax(hist)
print(np.max(hist), np.argmax(hist))

tolerance=10
lower_gray = np.array([elem-tolerance, 0,0])
upper_gray = np.array([elem+tolerance,255,255])
# Threshold the image to get only selected
mask = cv2.inRange(HSV_img, lower_gray, upper_gray)
# Bitwise-AND mask and original image
res2 = cv2.bitwise_and(im,im, mask= mask)

cv2.imshow('res',res)
cv2.waitKey()

titles = ['Original Image', 'Selected Gray Values', 'Hue', 'Result']
images = [img, res, hue, res2]
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()