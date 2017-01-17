# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 08:28:12 2017

@author: SzMike
"""

import os
import numpy as np
from skimage import morphology
from skimage import feature
from skimage import measure

import cv2
from defPaths import *
import tools

class parameters:
    pixelSize=1 # in microns
    magnification=1
    rbcR=25

if __name__ == '__main__':
    
    image_file=os.path.join(image_dir,'55.bmp')
    im = cv2.imread(image_file,cv2.IMREAD_COLOR)
    
    # choose best color channel - for separating background
    im_onech = im[:,:,1];
    
    # background - foreground binarization
    # foreground : all cells
    th, foreground_mask = cv2.threshold(im_onech,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # processing for dtf
    
    r=int(parameters.rbcR/2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    
    foreground_mask_open=cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # filling convex holes
    
    background_mask=255-foreground_mask_open
    
    output = cv2.connectedComponentsWithStats(background_mask, 8, cv2.CV_32S)
    
    
    for i in range(output[0]):
        area=output[2][i][4]
        if area<parameters.rbcR*parameters.rbcR/5: 
            foreground_mask_open[output[1]==i]=255
    
    
    # use dtf to find markers for watershed
    dist_transform = cv2.distanceTransform(foreground_mask_open,cv2.DIST_L2,5)
    
    dist_transform[dist_transform<parameters.rbcR*0.5]=0
        
    # watershed
    r=int(parameters.rbcR/2)
    kernel = np.ones((r,r),np.uint8)
    
    local_maxi = feature.peak_local_max(dist_transform, indices=False, footprint=np.ones((int(parameters.rbcR*0.6), int(parameters.rbcR*0.6))), labels=foreground_mask_open)
    local_maxi_dilate=cv2.dilate(local_maxi.astype('uint8')*255,kernel, iterations = 1)
    markers = measure.label(local_maxi_dilate)
    
    
    # watershed
    labels_ws = morphology.watershed(-dist_transform, markers, mask=foreground_mask_open)
    im2=im.copy()
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
        	cv2.circle(im2, (int(x), int(y)), int(r), (0, 255, 0), 2)
        	cv2.putText(im2, "#{}".format(label), (int(x) - 10, int(y)),
        		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
         
    cv2.imshow('alma',im2)
    cv2.waitKey()
    
    mag = tools.getGradientMagnitude(labels_ws.astype('float32'))
    mag[mag>0]=255

    tools.maskOverlay(im,mag,0.5,2,1)
    
