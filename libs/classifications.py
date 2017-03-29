# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:19:00 2017

@author: Szabolcs
"""

import numpy as np
from skimage import exposure

def crop_shape(im,mask,one_shape,rgb_norm,med_rgb,scale=1,adjust=True):
    # one_shape is a detected shape on im - it's elemnets are: (cell_type,polygon_type,pts,'None','None')
    mins=(np.min(one_shape[2],axis=0)*scale).astype('int32')
    maxs=(np.max(one_shape[2],axis=0)*scale).astype('int32')
    o=(mins+maxs)/2
    r=(maxs-mins)/2
    if min(mins)>=0 and maxs[1]<im.shape[0] and maxs[0]<im.shape[0]:
        im_cropped=im[max(mins[1],0):min(maxs[1],im.shape[0]-1),\
                            max(mins[0],0):min(maxs[0],im.shape[1]-1)]
        mask_cropped=mask[max(mins[1],0):min(maxs[1],im.shape[0]-1),\
                            max(mins[0],0):min(maxs[0],im.shape[1]-1)]
    
        if adjust:
            # Normalization
            # local to cropped
#            pixs=im_cropped[mask_cropped>0,]
#            nuc_med_rgb=np.median(pixs,axis=0)
            # global to image
            gamma=np.zeros(3)
            gain=np.zeros(3)
            for ch in range(3):
                gamma[ch]=np.log(255-rgb_norm[ch])/np.log(255-med_rgb[ch])
                gain[ch]=rgb_norm[ch]/np.power(med_rgb[ch],gamma[ch])
            im_cropped=exposure.adjust_gamma(im_cropped,np.mean(gamma),np.mean(gain))
   
    else:
        im_cropped=None
        mask_cropped=None
        
    return im_cropped, mask_cropped, o, r
        
 