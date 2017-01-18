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

image_file=os.path.join(image_dir,'82.bmp')

im = cv2.imread(image_file,cv2.IMREAD_COLOR)

# edge preserving filter???
#im2=cv2.edgePreservingFilter(im,im2,40,0.01)
#
#both = np.hstack((im,im2))
#
#cv2.imshow('seg',both)
#cv2.waitKey()
#cv2.destroyAllWindows()
# ToDo: check if RGB
# ToDo: rescale - have a standard image size
# convert to color space
im_cs = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

hist = tools.colorHist(im_cs,1)


# One channel image - alternatives: Green or luminance
im_onech = im_cs[:,:,1];
#im_onech = im_cs[:,:,2];

hist = tools.colorHist(im_onech,1)

# ToDo: noise filtering: median, Gaussian, bilateral

#im_denoise = cv2.bilateralFilter(im_onech,11,100,100)
im_denoise = cv2.GaussianBlur(im_onech,(4*int(parameters.rbcR/4)+1,4*int(parameters.rbcR/4)+1),4)
#im_denoise = cv2.medianBlur(im_eq,2*int(parameters.rbcR/4)+1)

hist = tools.colorHist(im_denoise,1)

both = np.hstack((im_onech,im_denoise))
cv2.imshow('seg',both)
cv2.waitKey()
cv2.destroyAllWindows()
# histogram
# ToDo: check contrast - enhance if needed
#im_eq = equ = cv2.equalizeHist(im_onech);

im_denoise = cv2.GaussianBlur(im_onech,(4*int(parameters.rbcR/4)+1,4*int(parameters.rbcR/4)+1),4)


# egyenetlen megvilágításnál
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(int(parameters.rbcR),int(parameters.rbcR)))
im_eq = clahe.apply(im_denoise)

#hist = tools.colorHist(im_eq,1)


hist = tools.colorHist(im_eq,1)

#plt.subplot(2,2,1),plt.imshow(im_onech,'gray')

# binarization
r=int(parameters.rbcR/2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))


th, sure_fg = cv2.threshold(im_eq,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
sure_fg=cv2.erode(sure_fg, kernel, iterations = 1)


tools.maskOverlay(im_onech,sure_fg,0.5,1)

#th, sure_fg_2 = cv2.threshold(im_eq,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#sure_fg_2=cv2.morphologyEx(bmask, cv2.MORPH_OPEN, kernel)

#tools.maskOverlay(im_onech,bmask2,0.5,1)


#plt.subplot(1,1,1),plt.imshow(mask,'gray')



# find seeds and segment cells

#kernel = np.ones((5,5),np.uint8)
#opening = cv2.morphologyEx(bmask2,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
#sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area



dist_transform = cv2.distanceTransform(sure_fg,cv2.DIST_L2,5)

dist_transform[dist_transform<10]=0

kernel = np.ones((3,3),np.uint8)

local_maxi = feature.peak_local_max(dist_transform, indices=False, footprint=np.ones((11, 11)), labels=bmask2)
local_maxi_dilate=cv2.dilate(local_maxi.astype('uint8')*255,kernel, iterations = 1)
markers = measure.label(local_maxi_dilate)
labels_ws = morphology.watershed(-dist_transform, markers, mask=bmask2)

tools.maskOverlay(im_onech,local_maxi_dilate,0.5,1)


#ret, sure_fg = cv2.threshold(dist_transform,20,255,0)
# Finding unknown region
#sure_fg = np.uint8(sure_fg)
    
#unknown = cv2.subtract(sure_bg,sure_fg)
   # Marker labelling
#ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
#    markers = markers+1
    # Now, mark the region of unknown with zero
#    markers[unknown==255] = 0

#distance = ndimage.distance_transform_edt(opening)
#local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((11, 11)), labels=bmask2)
#markers = measure.label(local_maxi,neighbors=True)


#labels_ws = morphology.watershed(-distance, markers, mask=mask)
#markers[~mask] = -1
#labels_rw = segmentation.random_walker(mask, markers)
#properties = measure.regionprops(img_mark)


ws=(labels_ws).astype('uint8')
wsC = cv2.applyColorMap(ws, cv2.COLORMAP_PARULA)

#
edges = cv2.Canny(ws,1,0)

im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im, contours, -1, (0,255,0), 3)

#m=ws==10
#m=255*m.astype('uint8')

tools.maskOverlay(im,im2,0.5,1)


cv2.namedWindow('alma')
cv2.imshow('alma',im)
cv2.waitKey()
cv2.destroyAllWindows()


x,y,w,h = cv2.boundingRect(contours)
img = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)