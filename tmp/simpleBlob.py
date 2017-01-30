# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:58:52 2017

@author: SzMike
"""

# Standard imports
import cv2
import os
import numpy as np;
import _init_path
import math

import matplotlib.pyplot as plt
from params import param
import tools
%matplotlib qt5
 
fi=plt.figure('image')
axi=fi.add_subplot(111)
fh=plt.figure('histogram')
axh=fh.add_subplot(111)

param=param()

imDirs=os.listdir(param.getImageDirs(''))
print(imDirs)
image_dir=param.getImageDirs(imDirs[0])
image_file=os.path.join(image_dir,'10.bmp')
im = cv2.imread(image_file,cv2.IMREAD_COLOR)
scale=1 #256/float(max(rgb.shape)) 
im_s=cv2.resize(im, (int(scale*im.shape[1]),int(scale*im.shape[0])), interpolation = cv2.INTER_AREA)

rgb = cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB)
im_cs = cv2.cvtColor(im_s, cv2.COLOR_BGR2HSV)

im_s2=im_cs
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
params.minArea = math.pow(0.5*param.rbcR*scale,2)*3.14
params.maxArea = math.pow(1*param.rbcR*scale,2)*3.14
params.filterByArea = True
params.thresholdStep=1
params.minConvexity=0.9
detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(im_s2)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im_s2, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 


r=np.empty(len(keypoints))
pt=[]
for i,item in enumerate(keypoints):
    r[i]=item.size/2
    pt.append(item.pt)
    x1=int(pt[i][0]-r[i])
    y1=int(pt[i][1]-r[i])
    x2=int(pt[i][0]+r[i])
    y2=int(pt[i][1]+r[i])
    cv2.rectangle(im_with_keypoints,(x1,y1),(x2,y2),(255,255,255),2)
    cv2.putText(im_with_keypoints, "#{}".format(i), (x1 - 10, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
axi.imshow(im_with_keypoints)
   
ext=1.25
i=0
x1=int(pt[1][0]-r[i])
y1=int(pt[i][1]-r[i])
x2=int(pt[i][0]+r[i])
y2=int(pt[i][1]+r[i])
im_cropped=im_s2[y1:y2,x1:x2]
    
axi.imshow(im_cropped)
hist = tools.colorHist(im_cropped,1)
