# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:09:21 2017

@author: SzMike
"""
from skimage import morphology
from skimage import feature
from skimage import measure

import numpy as np
import matplotlib.pyplot as plt
 
import segmentations
import imtools

def wbc_nucleus_mask(hsv,param,scale=1,sat_tsh=100,vis_diag=False,fig=''):
 
# create segmentation for WBC detection based on hue and saturation
# fix hue range is used
    mask_nuc=np.logical_and(np.logical_and(hsv[:,:,0]>param.wbc_range_in_hue[0]*255,\
                                            hsv[:,:,0]<param.wbc_range_in_hue[1]*255),\
                                            hsv[:,:,1]>sat_tsh) 
    
# morphological opening to eliminate small detections    
    mask_nuc=morphology.binary_opening(mask_nuc,morphology.disk(np.round(max(hsv.shape)/200)))
#    mask_fg=morphology.binary_closing(mask_fg,morphology.disk(np.round(param.middle_size/128)))
    
    return mask_nuc


def wbc_regions(mask_nuc,param,scale=1,vis_diag=False,fig=''):
    
    label_nuc = measure.label(mask_nuc, connectivity=mask_nuc.ndim)
       
    props = measure.regionprops(label_nuc) 

# 1st. step: remove small and boundary regions
    label_wbc=np.zeros(label_nuc.shape).astype('int32')
    props_large=props.copy()
    i_clean=0
    for ip, p in enumerate(props):
        if p.area<0.1*param.rbcR**2*np.pi*scale**2:
            props_large.remove(p)
            continue
        i_clean+=1
        label_wbc[label_nuc==ip+1]=i_clean
                 
#    props_sorted=sorted(props_large, key=lambda x: x.area)

# Merge small neighbouring areas    
    if len(props_large)>0:                   
        po=np.asarray([p.centroid for p in props_large])
        cent_dist=segmentations.center_diff_matrix(po,metric='euclidean')
      
        areas=[p.area for p in props_large]
        
        re_labels=np.linspace(0,len(props_large)-1,len(props_large))
        
        # merging regions
        # TODO: handle 2 thresholds: distance and size
    
        for i, row in enumerate(cent_dist):
            row_use=row[i+1:]
            if len(row_use)>0:
                m_i=np.argmin(row_use)
                if row_use[m_i]<3*param.rbcR*scale and\
                   (areas[i]+areas[m_i+i+1]<2*param.rbcR**2*np.pi*scale**2):
                   re_labels[re_labels==re_labels[m_i+i+1]]=re_labels[i]
                   areas[m_i+i+1]+=areas[i]
     
        re_label_wbc=label_wbc.copy()          
        for i, re_lab in enumerate(re_labels):   
            re_label_wbc[label_wbc==i+1]=re_lab+1
                        
    else:
        re_label_wbc=label_wbc

    prop_wbc=measure.regionprops(re_label_wbc)
    
    if vis_diag:
        fig=plt.figure('wbc nucleus labels',figsize=(7,7))
        axs=fig.add_subplot(111)
        axs.imshow(re_label_wbc)  
    
    # final removal of small elements
    for p in prop_wbc:
        if p.major_axis_length<0.75*param.rbcR:
            prop_wbc.remove(p)
        if p.major_axis_length>5*param.rbcR:
            prop_wbc.remove(p)
    return prop_wbc

def cell_mask(hsv,param,mask=None,scale=1,init_centers='k-means++',vis_diag=False,fig=''):
# Clustering in sv space - using the init_centers from clustering in smaller image     
    clust_centers_0, label_0 = segmentations.segment_hsv(hsv, init_centers=init_centers,\
                                                         chs=(1,1,2),\
                                                         n_clusters=5,\
                                                         vis_diag=vis_diag)   
    
    cent_dist=segmentations.center_diff_matrix(clust_centers_0,metric='euclidean')
    
    # adding meaningful labels
    ind_sat=np.argsort(clust_centers_0[:,0])
    ind_val=np.argsort(clust_centers_0[:,2])
    
    label_fg_bg=np.zeros(label_0.shape).astype('uint8')
    label_fg_bg[label_0==ind_val[-2]]=2 # unsure region
    label_fg_bg[label_0==ind_sat[-3]]=2 # unsure region
    label_fg_bg[label_0==ind_val[-1]]=1 # sure background
    label_fg_bg[label_0==ind_sat[-1]]=2 # unsure region
    label_fg_bg[label_0==ind_sat[-1]]=3 # cell foreground guess 1 
    label_fg_bg[label_0==ind_sat[-2]]=3 # cell foreground guess 2
    if cent_dist[ind_sat[-2],ind_sat[-3]]<cent_dist[ind_sat[-3],ind_val[-1]]:                 
       label_fg_bg[label_0==ind_sat[-3]]=3 # cell foreground guess 3
 
    if not mask is None:
        label_fg_bg[mask]=2         
# DO morphology on labels
    mask_cell=cell_mask_morphology(hsv,param,label_fg_bg,label_tsh=2,\
                                       scale=scale,vis_diag=vis_diag,fig='cell')    

    return mask_cell

def cell_mask_morphology(hsv,param,label_mask,label_tsh=2,scale=1,vis_diag=False,fig=''):
    
    mask_fg=label_mask>label_tsh
    mask_fg_open=mask_fg   
    mask_fg_filled=morphology.remove_small_holes(mask_fg_open>0, 
                                                 min_size=scale*param.cellFillAreaPct*param.rbcR*param.rbcR*np.pi, 
                                                 connectivity=2)
    mask_fg_clear=morphology.binary_opening(mask_fg_filled,morphology.disk(scale*param.rbcR*param.cellOpeningPct))

    if vis_diag:
        f=plt.figure(fig+'_morphology')

        ax0=f.add_subplot(221)
        imtools.normalize(label_mask,ax=ax0,vis_diag=vis_diag)
        ax0.set_title('label')

        ax1=f.add_subplot(222)
        imtools.maskOverlay(hsv,255*(mask_fg_open>0),0.5,ax=ax1,vis_diag=vis_diag)
        ax1.set_title('foreground') 

        ax2=f.add_subplot(223)
        imtools.maskOverlay(hsv,255*(mask_fg_filled>0),0.5,ax=ax2,vis_diag=vis_diag)
        ax2.set_title('filled')
                       
        ax3=f.add_subplot(224)
        imtools.maskOverlay(hsv,255*(mask_fg_clear>0),0.5,ax=ax3,vis_diag=vis_diag)
        ax3.set_title('clear')
        
    return mask_fg_clear

def cell_markers_from_mask(mask_cell,param,scale=1,vis_diag=False,fig=''):

    # use dtf to find markers for watershed 
    skel, dtf = morphology.medial_axis(mask_cell, return_distance=True)
    dtf.flat[(mask_cell>0).flatten()]+=np.random.random(((mask_cell>0).sum()))
    # watershed seeds
    # TODO - add parameters to cfg
    local_maxi = feature.peak_local_max(dtf, indices=False, 
                                        threshold_abs=0.25*param.rbcR*scale,
                                        footprint=np.ones((int(1.5*param.rbcR*scale), int(1.5*param.rbcR*scale))), 
                                        labels=mask_cell.copy())
    #markers, n_RBC = measure.label(local_maxi,return_num=True)
    markers=morphology.binary_dilation(local_maxi>0,morphology.disk(3))
    
    label= measure.label(markers, connectivity=markers.ndim)
    prop=measure.regionprops(label)
  
    return markers, prop