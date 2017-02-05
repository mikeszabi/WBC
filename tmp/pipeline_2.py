# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""

import _init_path
import os
import glob

import cv2
import skimage.io as io
import numpy as np;
import math
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
%matplotlib qt5
import cfg
import tools
import segment_kmeans
import annotations
 
##
param=cfg.param()
vis_diag=True

imDirs=os.listdir(param.getImageDirs(''))
print(imDirs)
image_dir=param.getImageDirs(imDirs[1])
print(glob.glob(os.path.join(image_dir,'*.bmp')))
image_file=os.path.join(image_dir,'94.bmp')

#
im = io.imread(image_file) # read uint8 image

# rescale if needed
#im, scale = tools.imresizeMaxDim(im, max(im.shape))

# im from bgr
if vis_diag:
    fo=plt.figure('original image')
    axo=fo.add_subplot(111)
    axo.imshow(im)

cent, lab = segment_kmeans.segment(im, vis_diag=True)

# lab == 0 : overexposed
# lab == 1 : sure bckg
# lab ==2 : unsure
# lab == 3 : sure foreground

bg_mask= np.logical_or(lab == 0, lab == 1)
im_illinhom= segment_kmeans.illumination_inhomogenity(im, bg_mask, vis_diag=True)
  
# TODO: take sure foreground - clean and fill the holes
sure_fg_mask=((lab==3)*255).astype('uint8')
tools.maskOverlay(im,sure_fg_mask,0.5,vis_diag=True,fig='sure_fg_mask')

unsure_mask=((lab==2)*255).astype('uint8')
output = cv2.connectedComponentsWithStats(255-sure_fg_mask, 8, cv2.CV_32S,connectivity=4)
lab=output[1]
tools.normalize(lab,vis_diag=True,fig='holes')
for i in range(output[0]):
    area=max(output[2][i][4],1)
    label_mask=output[1]==i
    unsure_sum=unsure_mask.flat[label_mask.flatten()].sum()/255
    if area<param.rbcR*param.rbcR*np.pi and unsure_sum/area>=0: #cv2.isContourConvex(:
        # ToDo and if convex
        print(i)
        sure_fg_mask[output[1]==i]=255
tools.maskOverlay(im,sure_fg_mask,0.5,vis_diag=True,fig='sure_fg_mask_filled')

# opening
r=int(param.rbcR)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))

fg_mask_open=cv2.morphologyEx(sure_fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

tools.maskOverlay(im,fg_mask_open,0.5,vis_diag=True)


 # use dtf to find markers for watershed
dist_transform = cv2.distanceTransform(fg_mask_open,cv2.DIST_L2,5)

# remove small blobs
dist_transform[dist_transform<param.rbcR*0.5]=0
    
# watershed
r=int(0.75*param.rbcR)
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

# ToDo: merge groups
shapelist=[]
for label in np.unique(labels_ws):
    	# if the label is zero, we are examining the 'background'
    	# so simply ignore it
     if label == 0:
         continue
  
     mask = np.zeros(bg_mask.shape, dtype="uint8")
     mask[labels_ws == label] = 255
     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
     c = max(cnts, key=cv2.contourArea)
     one_shape=('RBC','general',c.reshape(c.shape[0],c.shape[2]),'None','None')
     shapelist.append(one_shape)
#     x,y,w,h = cv2.boundingRect(c)
     #TODO: túl nagy pacnik további vizsgálata
#     if ((x>param.rbcR) & (x+w<im.shape[1]-param.rbcR) & 
#         (y>param.rbcR) & (y+h<im.shape[0]-param.rbcR)):
#        cv2.rectangle(im2,(x,y),(x+w,y+h),(255,255,255),2)
#        cv2.putText(im2, "#{}".format(label), (x - 10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
#        if cv2.contourArea(c)>2*int(math.pi*math.pow(param.wbcRatio*param.rbcR,2)):
#            cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),3)
fh=plt.figure('detected')
ax=fh.add_subplot(111)
ax.imshow(im2)

head, tail=os.path.split(image_file)
tmp = annotations.AnnotationWriter(head,tail.replace('.bmp',''), (im.shape[0],im.shape[1]))
tmp.addShapes(shapelist)
tmp.save()