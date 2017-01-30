# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:55:57 2017

@author: SzMike
"""

import os
import _init_path
import numpy as np
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
import math

import cv2

import matplotlib.pyplot as plt
%matplotlib qt5

from params import param
import tools

param=param()
    
imDirs=os.listdir(param.getImageDirs(''))
print(imDirs)
image_dir=param.getImageDirs(imDirs[0])
image_file=os.path.join(image_dir,'10.bmp')
im = cv2.imread(image_file,cv2.IMREAD_COLOR)


scale= 1280/float(max(im.shape[0],rgb.shape[1]))   
im_small=cv2.resize(im, (int(scale*im.shape[1]),int(scale*im.shape[0])), interpolation = cv2.INTER_AREA)
im_small=cv2.cvtColor(im_small, cv2.COLOR_BGR2RGB)
#plt.imshow(im,hold=False)
#hist = tools.colorHist(im,1)

im_cs = cv2.cvtColor(im_small, cv2.COLOR_BGR2HSV)


#hist = tools.colorHist(im_cs,1)

# KMEANS
#Z = im_cs.reshape((-1,3))
#Z = np.float32(Z)/256
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#K = 4
#ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#center = np.uint8(center*256)
#res = center[label.flatten()]
#res2 = res.reshape((im.shape))
#
#cv2.imshow('res2',res2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# choose best color channel - for separating background
#im_onech = im_small[:,:,1];
im_onech = im_cs[:,:,1];             
             
#cC = cv2.applyColorMap(im_onech, cv2.COLORMAP_JET)     
     
#hist = tools.colorHist(im_onech,1)
#plt.imshow(im_onech)

# homogen illumination correction
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
#im_eq = clahe.apply(im_onech)
im_eq=im_onech
#plt.imshow(im_eq)
#hist = tools.colorHist(im_eq,1)


#im_denoise = cv2.GaussianBlur(im_eq,(4*int(param.rbcR/10)+1,4*int(param.rbcR/10)+1),4)
#im_denoise = cv2.bilateralFilter(im_eq,param.rbcR,10,param.rbcR)
#http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html
#plt.imshow(im_denoise)
#hist = tools.colorHist(im_denoise,1)

#im_denoise=im_onech
# background - foreground binarization
# foreground : all cells
th, foreground_mask = cv2.threshold(im_eq,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#tools.maskOverlay(im_onech,foreground_mask,0.5,2,1)

#mask_filled=tools.floodFill(foreground_mask)
#tools.maskOverlay(im_onech,mask_filled,0.5,2,1)

# processing for dtf

r=int(1.2*param.rbcR)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))

foreground_mask_open=cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)

#tools.maskOverlay(im_onech,foreground_mask_open,0.5,2,1)

# filling convex holes

background_mask=255-foreground_mask_open
#edges = cv2.Canny(foreground_mask_open,1,0)
#tools.maskOverlay(im,background_mask,0.5,2,1)

output = cv2.connectedComponentsWithStats(background_mask, 8, cv2.CV_32S)
lab=output[1]
tools.normalize(lab,0)


for i in range(output[0]):
    area=output[2][i][4]
    if area<param.rbcR*param.rbcR/5: #cv2.isContourConvex(:
        # ToDo and if convex
        print(i)
        foreground_mask_open[output[1]==i]=255
# TODO: FILL holes in cells - i.e. convex holes!

#tools.maskOverlay(im,foreground_mask_open,0.5,2,1)


# use dtf to find markers for watershed
dist_transform = cv2.distanceTransform(foreground_mask_open,cv2.DIST_L2,5)

dist_transform[dist_transform<param.rbcR*0.5]=0
    
# watershed
r=int(param.rbcR)
kernel = np.ones((r,r),np.uint8)

local_maxi = feature.peak_local_max(dist_transform, indices=False, footprint=np.ones((int(param.rbcR*0.6), int(param.rbcR*0.6))), labels=foreground_mask_open)
local_maxi_dilate=cv2.dilate(local_maxi.astype('uint8')*255,kernel, iterations = 1)
markers = measure.label(local_maxi_dilate)


# watershed on dtf
labels_ws = morphology.watershed(-dist_transform, markers, mask=foreground_mask_open)

# edge map for visualization
mag=segmentation.find_boundaries(labels_ws).astype('uint8')*255

im2=tools.maskOverlay(im_small,mag,0.5,1,1)
# counting

for label in np.unique(labels_ws):
    	# if the label is zero, we are examining the 'background'
    	# so simply ignore it
     if label == 0:
         continue
  
     mask = np.zeros(im_onech.shape, dtype="uint8")
     mask[labels_ws == label] = 255
     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
     c = max(cnts, key=cv2.contourArea)
     x,y,w,h = cv2.boundingRect(c)
     #if ((x>param.rbcR) & (x+w<im.shape[1]-param.rbcR) & 
         #(y>param.rbcR) & (y+h<im.shape[0]-param.rbcR)):
        #cv2.rectangle(im2,(x,y),(x+w,y+h),(255,255,255),2)
        #cv2.putText(im2, "#{}".format(label), (x - 10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
        #if cv2.contourArea(c)>2*int(math.pi*math.pow(param.wbcRatio*param.rbcR,2)):
            #cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),3)
            
plt.imshow(im2)
