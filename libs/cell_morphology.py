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
import segmentations


def rbc_labels(im,clust_centers_0,label_0,vis_diag=False):
    # creating meaningful labels for foreground-background segmentation and RBC detection
    cent_dist=segmentations.center_diff_matrix(clust_centers_0,metric='euclidean')
    
    # adding meaningful labels
    ind_sat=np.argsort(clust_centers_0[:,0])
    ind_val=np.argsort(clust_centers_0[:,2])
    
    label_fg_bg=np.zeros(label_0.shape).astype('uint8')
    label_fg_bg[label_0==ind_val[-2]]=2 # unsure region
    label_fg_bg[label_0==ind_sat[-3]]=2 # unsure region
    label_fg_bg[label_0==ind_val[-1]]=1 # sure background
    label_fg_bg[label_0==ind_sat[-1]]=31 # cell foreground guess 1 
    if cent_dist[ind_sat[-1],ind_sat[-2]]<cent_dist[ind_sat[-2],ind_val[-1]]:
       label_fg_bg[label_0==ind_sat[-2]]=32 # cell foreground guess 2
       if cent_dist[ind_sat[-2],ind_sat[-3]]<cent_dist[ind_sat[-3],ind_val[-1]]:                 
           label_fg_bg[label_0==ind_sat[-3]]=33 # cell foreground guess 3
          
    return label_fg_bg


def rbc_mask_morphology(im,label_mask,param,label_tsh=3,scale=1,vis_diag=False,fig=''):
    
    mask_fg=label_mask>label_tsh
    mask_fg_open=morphology.binary_opening(mask_fg,morphology.star(2))
#   
#    mask_fg=label_mask==32
#    mask_fg_open_2=morphology.binary_closing(mask_fg,morphology.disk(1)).astype('uint8')
#    mask_fg=np.logical_or(mask_fg_open_1,mask_fg_open_2)
#   
    mask_fg_filled=morphology.remove_small_holes(mask_fg_open>0, 
                                                 min_size=scale*param.cellFillAreaPct*param.rbcR*param.rbcR*np.pi, 
                                                 connectivity=2)
    mask_fg_clear=morphology.binary_opening(mask_fg_filled,morphology.disk(scale*param.rbcR*param.cellOpeningPct)).astype('uint8')

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
   
def rbc_markers_from_mask(mask_fg_clear,param):

    # use dtf to find markers for watershed 
    skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
    dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
    # watershed seeds
    # TODO - add parameters to cfg
    local_maxi = feature.peak_local_max(dtf, indices=False, 
                                        threshold_abs=0.5*param.rbcR,
                                        footprint=np.ones((int(1.5*param.rbcR), int(1.5*param.rbcR))), 
                                        labels=mask_fg_clear.copy())
    markers, n_RBC = measure.label(local_maxi,return_num=True)
    segmentation.clear_border(markers,buffer_size=50,in_place=True)
    
    return markers

#def wbc_masks(label_1, clust_sat,param,scale,vis_diag=False):
#    # creating masks for labels        
#    n=np.zeros(clust_sat.shape[0])
#    mask_tmps=[]
#    for i, c in enumerate(clust_sat):
#        mask_tmp=morphology.binary_opening(label_1==i,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR)))
#        mask_tmps.append(mask_tmp)
#        n[i]=mask_tmp.sum()
##        
#    ind_n=np.argsort(n)
#    ind_sat=np.argsort(clust_sat)
#   
#    if n[ind_n[-1]]/sum(n)>param.over_saturated_rbc_ratio:
#        if ind_sat[-1]==ind_n[-1]:
## normally pixels with highest saturations are candidates for wbc segments, but here rbc-s have higher sat
## TODO: add diagnostics
#            mask_pot_wbc_1=mask_tmps[ind_sat[-2]]
#            mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-3]],mask_tmps[ind_sat[-2]])
#            print('undersaturated wbc') # wbc is not at highest saturation
#        elif ind_sat[-2]==ind_n[-1]:
#            mask_pot_wbc_1=mask_tmps[ind_sat[-1]]
#            mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-3]],mask_tmps[ind_sat[-1]])
#        else:
#            mask_pot_wbc_1=mask_tmps[ind_sat[-1]]
#            mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-2]],mask_tmps[ind_sat[-1]]) 
#    else:
#        mask_pot_wbc_1=mask_tmps[ind_sat[-1]]
#        mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-2]],mask_tmps[ind_sat[-1]]) 
#        
#    mask_wbc_pot=[]    
## TODO: dd parameters
#    mask_wbc_pot.append(morphology.binary_opening(mask_pot_wbc_1,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR))))
#    mask_wbc_pot.append(morphology.binary_opening(mask_pot_wbc_2,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR))))
#
#    return mask_wbc_pot