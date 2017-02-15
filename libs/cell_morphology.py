# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:48:17 2017

@author: SzMike
"""

from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
import numpy as np
import matplotlib.pyplot as plt

import imtools
import cfg

def rbc_mask_morphology(im,label_mask,label_tsh=3,vis_diag=False,fig=''):
    
    param=cfg.param()
    mask_fg=label_mask>label_tsh
    mask_fg_open=morphology.binary_opening(mask_fg,morphology.star(2))
#   
#    mask_fg=label_mask==32
#    mask_fg_open_2=morphology.binary_closing(mask_fg,morphology.disk(1)).astype('uint8')
#    mask_fg=np.logical_or(mask_fg_open_1,mask_fg_open_2)
#   
    mask_fg_filled=morphology.remove_small_holes(mask_fg_open>0, 
                                                 min_size=param.cellFillAreaPct*param.rbcR*param.rbcR*np.pi, 
                                                 connectivity=2)
    mask_fg_clear=morphology.binary_opening(mask_fg_filled,morphology.disk(param.rbcR*param.cellOpeningPct)).astype('uint8')

    if vis_diag:
        f=plt.figure(fig+'_cell_overlayed')

        ax0=f.add_subplot(221)
        imtools.normalize(label_mask,ax=ax0,vis_diag=vis_diag)
        ax0.set_title('label')

        ax1=f.add_subplot(222)
        imtools.maskOverlay(im,255*(mask_fg_open>0),0.5,ax=ax1,vis_diag=vis_diag)
        ax1.set_title('foreground') 

        ax2=f.add_subplot(223)
        imtools.maskOverlay(im,255*(mask_fg_filled>0),0.5,ax=ax2,vis_diag=vis_diag)
        ax2.set_title('filled')
                       
        ax3=f.add_subplot(224)
        imtools.maskOverlay(im,255*(mask_fg_clear>0),0.5,ax=ax3,vis_diag=vis_diag)
        ax3.set_title('clear')
        
    return mask_fg_clear
   
def rbc_markers_from_mask(mask_fg_clear):
    param=cfg.param()

    # use dtf to find markers for watershed 
    skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
    dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
    # watershed seeds
    # TODO - add parameters to cfg
    local_maxi = feature.peak_local_max(dtf, indices=False, 
                                        threshold_abs=0.25*param.rbcR,
                                        footprint=np.ones((int(1.5*param.rbcR), int(1.5*param.rbcR))), 
                                        labels=mask_fg_clear.copy())
    markers, n_RBC = measure.label(local_maxi,return_num=True)
    segmentation.clear_border(markers,buffer_size=50,in_place=True)
    
    return markers