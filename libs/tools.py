# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:57:09 2017

@author: SzMike
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# colorhits
def colorHist(im,plotFlag=False,ax='none'):
    color = ('r','g','b')
    histr=[]
    if len(im.shape)==2:
        nCh=1
    else:
        nCh=3    
    for i in range(nCh):
        histr.append(cv2.calcHist([im],[i],None,[256],[0,256]))
        if plotFlag:
            if ax=='none':
                fh=plt.figure('histogram')
                ax=fh.add_subplot(111)
            ax.plot(histr[i],color = color[i])
            ax.set_xlim([0,256])
    return histr
    
def maskOverlay(im,mask,alpha,ch,sbs=False,plotFlag=False,ax='none'):
# mask is 2D binary
# image can be 1 or 3 channel
# sbs: side by side
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
        if sbs:
            both = np.hstack((im_3,im_overlay))
        else:
            both=im_overlay
        if ax=='none':
            fi=plt.figure('overlayed')
            ax=fi.add_subplot(111)
        ax.imshow(both)
    return im_overlay
    
def normalize(im,plotFlag=False,ax='none'):
    im_norm=cv2.normalize(im, 0, 255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    if plotFlag:
        if ax=='none':
            fi=plt.figure('normalized')
            ax=fi.add_subplot(111)
        ax.imshow(im_norm)
    return im_norm
          
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

def imresize(img, scale, interpolation = cv2.INTER_LINEAR):
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)


def imresizeMaxDim(img, maxDim, boUpscale = False, interpolation = cv2.INTER_LINEAR):
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1  or boUpscale:
        img = imresize(img, scale, interpolation)
    else:
        scale = 1.0
    return img, scale