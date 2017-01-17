# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:55:57 2017

@author: SzMike
"""

import os
#import importlib
import numpy as np
#from skimage import segmentation
#from scipy import ndimage
from skimage import morphology
from skimage import feature
#from skimage import restoration
from skimage import measure

import cv2
#from matplotlib import pyplot as plt
from defPaths import *
import tools

class parameters:
    pixelSize=1 # in microns
    magnification=1
    rbcR=25

## read image    
image_file=os.path.join(image_dir,'60.bmp')
im = cv2.imread(image_file,cv2.IMREAD_COLOR)

# choose best color channel - for separating background
im_onech = im[:,:,1];
hist = tools.colorHist(im_onech,1)

# homogen illumination correction
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#im_eq = clahe.apply(im_onech)

#hist = tools.colorHist(im_eq,1)

# background - foreground binarization
# foreground : all cells
th, foreground_mask = cv2.threshold(im_onech,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

tools.maskOverlay(im_onech,foreground_mask,0.5,2,1)

# processing for dtf

r=int(parameters.rbcR/2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))

foreground_mask_open=cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)

tools.maskOverlay(im_eq,foreground_mask_open,0.5,2,1)

# filling convex holes

background_mask=255-foreground_mask_open
#edges = cv2.Canny(foreground_mask_open,1,0)
tools.maskOverlay(im,background_mask,0.5,2,1)

output = cv2.connectedComponentsWithStats(background_mask, 8, cv2.CV_32S)
lab=output[1]
tools.normalize(lab,1)


for i in range(output[0]):
    area=output[2][i][4]
    if area<parameters.rbcR*parameters.rbcR/5: #cv2.isContourConvex(:
        # ToDo and if convex
        print(i)
        foreground_mask_open[output[1]==i]=255
# TODO: FILL holes in cells - i.e. convex holes!

tools.maskOverlay(im,foreground_mask_open,0.5,2,1)


# use dtf to find markers for watershed
dist_transform = cv2.distanceTransform(foreground_mask_open,cv2.DIST_L2,5)

dist_transform[dist_transform<parameters.rbcR/2]=0

tools.normalize(dist_transform,1)


kernel = np.ones((9,9),np.uint8)

local_maxi = feature.peak_local_max(dist_transform, indices=False, footprint=np.ones((int(parameters.rbcR*0.6), int(parameters.rbcR*0.6))), labels=foreground_mask_open)
local_maxi_dilate=cv2.dilate(local_maxi.astype('uint8')*255,kernel, iterations = 1)
markers = measure.label(local_maxi_dilate)

tools.maskOverlay(im_onech,local_maxi_dilate,0.5,1,1)


# watershed
labels_ws = morphology.watershed(-dist_transform, markers, mask=foreground_mask_open)

for label in np.unique(labels_ws):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
 
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(im_onech.shape, dtype="uint8")
	mask[labels_ws == label] = 255
 
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(cnts, key=cv2.contourArea)
 
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(im, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(im, "#{}".format(label), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
cv2.imshow('alma',im)
cv2.waitKey()

mag = tools.getGradientMagnitude(labels_ws.astype('float32'))
mag[mag>0]=255

tools.maskOverlay(im,mag,0.5,2,1)


#wsC = cv2.applyColorMap(labels_ws, cv2.COLORMAP_PARULA)

#    cv2.namedWindow('alma')
cv2.imshow('alma',mag)
cv2.waitKey()
#    cv2.destroyAllWindows()

im2, contours, hierarchy = cv2.findContours(mag,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
kernel = np.ones((1,1),np.uint8)
im2=cv2.dilate(im2,kernel, iterations = 1)
im3=tools.maskOverlay(im,im2,0.5,1,1)

cv2.imshow('alma',mag)
cv2.waitKey()
#

for cnt in contours:
    rect = cv2. boundingRect(cnt)
    
#cv2.rectangle(im, rect, (0,0,255), 2)
