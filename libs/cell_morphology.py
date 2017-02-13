# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:48:17 2017

@author: SzMike
"""

from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
import imtools

def rbc_mask_morphology(im,label_mask,param,vis_diag=False,fig=''):
    
    mask_fg=label_mask>30
#    mask_fg_open_1=morphology.binary_closing(mask_fg,morphology.disk(1)).astype('uint8')
#   
#    mask_fg=label_mask==32
#    mask_fg_open_2=morphology.binary_closing(mask_fg,morphology.disk(1)).astype('uint8')
#    mask_fg=np.logical_or(mask_fg_open_1,mask_fg_open_2)
#   
    mask_fg_filled=morphology.remove_small_holes(mask_fg, 
                                                 min_size=param.cellFillAreaPct*param.rbcR*param.rbcR*np.pi, 
                                                 connectivity=0)
    mask_fg_clear=morphology.binary_opening(mask_fg_filled,morphology.disk(param.rbcR*param.cellOpeningPct)).astype('uint8')

    if vis_diag:
        f=plt.figure(fig+'_cell_overlayed')

        ax0=f.add_subplot(221)
        imtools.normalize(label_mask,ax=ax0,vis_diag=vis_diag)
        ax0.set_title('label')

        ax1=f.add_subplot(222)
        imtools.maskOverlay(im,255*(mask_fg>0),0.5,ax=ax1,vis_diag=vis_diag)
        ax1.set_title('foreground') 

        ax2=f.add_subplot(223)
        imtools.maskOverlay(im,255*(mask_fg_filled>0),0.5,ax=ax2,vis_diag=vis_diag)
        ax2.set_title('filled')
                       
        ax3=f.add_subplot(224)
        imtools.maskOverlay(im,255*(mask_fg_clear>0),0.5,ax=ax3,vis_diag=vis_diag)
        ax3.set_title('clear')
        
    return mask_fg_clear
   