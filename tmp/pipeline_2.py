# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""

import _init_path
import os
import glob

import cv2
import numpy as np;
import math
import matplotlib.pyplot as plt
%matplotlib qt5
import cfg
import tools
import segment_kmeans
 
##
param=cfg.param()

imDirs=os.listdir(param.getTestImageDirs(''))
print(imDirs)
image_dir=param.getTestImageDirs(imDirs[2])
print(glob.glob(os.path.join(image_dir,'*.bmp')))
image_file=os.path.join(image_dir,'60.bmp')

#
im = cv2.imread(image_file,cv2.IMREAD_COLOR)

# rescale
im_s, scale = tools.imresizeMaxDim(im, 1280)

rgb = cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB)
#fo=plt.figure('rgb')
#axo=fo.add_subplot(111)
#axo.imshow(rgb)

im_cs = cv2.cvtColor(im_s, cv2.COLOR_BGR2HSV)
im_s2=im_cs

center, masks = segment_kmeans.segment(im_s2, plotFlag=True)

#    masks[:,:,0]=overexpo_mask
#    masks[:,:,1]=sure_fg_mask
#    masks[:,:,2]=sure_bg_mask
#    masks[:,:,3]=unsure_mask

maxi=np.argmax(center[:,1])
mean_background_intensity=center[maxi,1]
illumination_inhomogenity= segment_kmeans.inhomogen(im_cs[:,:,2], masks[:,:,2], mean_background_intensity, plotFlag=True)
  

# TODO: take sure foreground - clean and fill the holes

sure_fg_mask=masks[:,:,1]
tools.maskOverlay(rgb,sure_fg_mask,0.5,plotFlag=True)

output = cv2.connectedComponentsWithStats(255-sure_fg_mask, 8, cv2.CV_32S)
lab=output[1]
tools.normalize(lab,plotFlag=True)
for i in range(output[0]):
    area=output[2][i][4]
    print(area)
    if area<param.rbcR*param.rbcR/2: #cv2.isContourConvex(:
        # ToDo and if convex
        print(i)
        sure_fg_mask[output[1]==i]=255
tools.maskOverlay(rgb,sure_fg_mask,0.5,plotFlag=True)

# opening
r=int(param.rbcR)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))

fg_mask_open=cv2.morphologyEx(sure_fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

tools.maskOverlay(rgb,fg_mask_open,0.5,plotFlag=True)


 # use dtf to find markers for watershed
dist_transform = cv2.distanceTransform(fg_mask_open,cv2.DIST_L2,5)

# remove small blobs
dist_transform[dist_transform<param.rbcR*0.5]=0
    
# watershed
r=int(0.5*param.rbcR)
kernel = np.ones((r,r),np.uint8)

local_maxi = feature.peak_local_max(dist_transform, indices=False, 
                                    footprint=np.ones((int(param.rbcR), int(param.rbcR))), labels=fg_mask_open)
# remove noisy maximas
local_maxi_dilate=cv2.dilate(local_maxi.astype('uint8')*255,kernel, iterations = 1)
markers = measure.label(local_maxi_dilate)


# watershed on dtf
labels_ws = morphology.watershed(-dist_transform, markers, mask=fg_mask_open)

# edge map for visualization
mag=segmentation.find_boundaries(labels_ws).astype('uint8')*255

im2=tools.maskOverlay(im,mag,0.5,1,0)
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
     #TODO: túl nagy pacnik további vizsgálata
#     if ((x>param.rbcR) & (x+w<im.shape[1]-param.rbcR) & 
#         (y>param.rbcR) & (y+h<im.shape[0]-param.rbcR)):
#        cv2.rectangle(im2,(x,y),(x+w,y+h),(255,255,255),2)
#        cv2.putText(im2, "#{}".format(label), (x - 10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
#        if cv2.contourArea(c)>2*int(math.pi*math.pow(param.wbcRatio*param.rbcR,2)):
#            cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),3)
plt.imshow(im2)