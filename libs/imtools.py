# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:57:09 2017

@author: SzMike
"""

import warnings
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.exposure import cumulative_distribution
from skimage import filters, img_as_ubyte
from mpl_toolkits.axes_grid1 import make_axes_locatable

# colorhist  - works for grayscale and color images
def colorHist(im,vis_diag=False,mask=None,fig=''):
    assert im.ndim==3, 'Not 3channel image'
    color = ('r','g','b')
    histr=[]
    if len(im.shape)==2:
        nCh=1
    else:
        nCh=3    
    for i in range(nCh):
        if mask==None:
            im_masked=im[:,:,i]
        else:
            im_masked=im[:,:,i].flat[mask.flatten()>0]
        h, b=np.histogram(im_masked,bins=range(255))
        histr.append(h)
        if vis_diag:
            fh=plt.figure(fig+'_histogram')
            ax=fh.add_subplot(111)
            ax.plot(histr[i],color = color[i])
            ax.set_xlim([0,256])
    return histr
    
def maskOverlay(im,mask,alpha,ch=1,sbs=False,vis_diag=False,fig=''):
# mask is 2D binary
# image can be 1 or 3 channel
# ch : rgb -> 012
# sbs: side by side
# http://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
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
    
    if vis_diag:
        if sbs:
            both = np.hstack((im_3,im_overlay))
        else:
            both=im_overlay
        fi=plt.figure(fig+'_overlayed')
        ax=fi.add_subplot(111)
        ax.imshow(both)
    return im_overlay
    
def normalize(im,vis_diag=False,fig=''):
    assert im.ndim==2, 'Not 1channel image'
    cdf, bins=cumulative_distribution(im, nbins=256)
    minI=bins[np.argwhere(cdf>0.01)[0,0]]
    maxI=bins[np.argwhere(cdf>0.99)[0,0]]
    im_norm=im.copy()
    im_norm[im_norm<minI]=minI
    im_norm[im_norm>maxI]=maxI
    im_norm=(im_norm-minI)/(maxI-minI)
    im_norm=(255*im_norm).astype('uint8')       
    if vis_diag:
        fi=plt.figure(fig+'_normalized')
        axi=fi.add_subplot(111)
        divider = make_axes_locatable(axi)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        i=axi.imshow(im_norm,cmap='jet')
        fi.colorbar(i, cax=cax, orientation='vertical')
        plt.show()
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
    #Get magnitude of gradient for given image"
    assert len(im.shape)==2, "Not 2D image"
    mag = filters.scharr(im)
    return mag


def imRescaleMaxDim(im, maxDim, boUpscale = False, interpolation = 1):
    scale = 1.0 * maxDim / max(im.shape[:2])
    if scale < 1  or boUpscale:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = img_as_ubyte(rescale(im, scale, order=interpolation))
    else:
        scale = 1.0
    return im, scale
