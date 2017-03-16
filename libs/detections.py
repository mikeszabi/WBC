# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:09:21 2017

@author: SzMike
"""
from skimage import morphology
import numpy as np;
 

def wbc_nucleus_mask(hsv,param,scale=1,sat_tsh=100,vis_diag=False,fig=''):
 
    """
    WBC nucleus masks
    """
# create segmentation for WBC detection based on hue and saturation
    label_wbc=np.logical_and(np.logical_and(hsv[:,:,0]>param.wbc_range_in_hue[0]*255,\
                                            hsv[:,:,0]<param.wbc_range_in_hue[1]*255),\
                                            hsv[:,:,1]>sat_tsh) 

    mask_fg=label_wbc>0
    mask_fg=morphology.binary_opening(mask_fg,morphology.disk(np.round(param.middle_size/256)))
#    mask_fg=morphology.binary_closing(mask_fg,morphology.disk(np.round(param.middle_size/128)))
    
    return mask_fg