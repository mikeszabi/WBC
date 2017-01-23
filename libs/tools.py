# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:57:09 2017

@author: SzMike
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# colorhits
def colorHist(im,plotFlag):
    color = ('r','g','b')
    histr=[]
    if len(im.shape)==2:
        nCh=1
    else:
        nCh=3    
    for i in range(nCh):
        histr.append(cv2.calcHist([im],[i],None,[256],[0,256]))
        if plotFlag:
            plt.plot(histr[i],color = color[i])
            plt.xlim([0,256])
            plt.show()
    return histr
    
def maskOverlay(im,mask,alpha,ch,plotFlag):
# mask is 2D binary
# image can be 1 or 3 channel
    if ch>2:
        ch=1

    mask_tmp=np.empty(mask.shape+(3,), dtype='uint8')   
    mask_tmp.fill(0)
    mask_tmp[:,:,ch]=mask
    
    if len(im.shape)==2:
        im_3=np.matlib.repeat(np.expand_dims(im,2),3,2)
    else:
        im_3=im
        
    im_overlay=cv2.addWeighted(mask_tmp,alpha,im_3,1-alpha,0) 
    
    if plotFlag:
        both = np.hstack((im_3,im_overlay))

        cv2.imshow('overlay',both)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return im_overlay
    
def normalize(im,plotFlag):
    im_norm=cv2.normalize(im, 0, 255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    if plotFlag:
        cv2.imshow('norm',im_norm)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return im_norm
        
def dtfSegment(mask):
    #im must be 1 channel
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,20,255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    
    tools.maskOverlay(mask,sure_fg,0.5,1)

    
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
          
def floodFill(mask):
    im_floodfill = mask.copy()
 
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = mask.shape[:2]
    mask_new = np.zeros((h+2, w+2), np.uint8)
 
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask_new, (0,0), 255);
 
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
    # Combine the two images to get the foreground.
    im_out = mask | im_floodfill_inv    
    return im_out      
    
def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

