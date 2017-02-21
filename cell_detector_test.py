# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:21:59 2017

@author: SzMike
"""
import os
import warnings
import skimage.io as io
import numpy as np;
from skimage.transform import resize
from skimage import morphology
from skimage import filters
from skimage import measure
from skimage import segmentation
from skimage.draw import polygon
from skimage import img_as_ubyte, img_as_float
import matplotlib.pyplot as plt
# %matplotlib qt5

import cfg
import imtools
import diagnostics
import segmentations
import cell_morphology
import annotations

def main(image_file,vis_diag=False):

    vis_diag=False

    # SET THE PARAMETERS
    param=cfg.param()
    output_dir=param.getOutDir('output')
    diag_dir=param.getOutDir('diag')
    
    # READ THE IMAGE
    im = io.imread(image_file) # read uint8 image
    # TODO: check image
                  
    # diagnose image, create overexpo mask and correct for inhomogen illumination
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
    diag.writeDiagnostics(diag_dir)   
    
    """
    Foreground and wbc segmentation
    """                   
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,param.middle_size,interpolation = 0)
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,param.middle_size,interpolation = 0)
 
    # create foreground mask using previously set init centers
    cent_2, label_2 = segmentations.segment_hsv(hsv_resize, init_centers=diag.cent_init, chs=(1,1,2), n_clusters=4, vis_diag=vis_diag)   
    cent_dist=segmentations.center_diff_matrix(cent_2,metric='euclidean')
    
    # adding meaningful labels
    ind_sat=np.argsort(cent_2[:,0])
    ind_val=np.argsort(cent_2[:,2])
    
    label_mask_resize=np.zeros(hsv_resize.shape[0:2]).astype('uint8')
    label_mask_resize[label_2==ind_val[-1]]=1 # sure background
    label_mask_resize[label_2==ind_val[-2]]=2 # unsure region
    label_mask_resize[label_2==ind_sat[-1]]=31 # sure cell foreground guess 1 
    if cent_dist[ind_sat[-1],ind_sat[-2]]<cent_dist[ind_sat[-2],ind_val[-1]]:
       label_mask_resize[label_2==ind_sat[-2]]=32 # sure cell foreground guess 2
       if cent_dist[ind_sat[-2],ind_sat[-3]]<cent_dist[ind_sat[-3],ind_val[-1]]:                 
           label_mask_resize[label_2==ind_sat[-3]]=33 # sure cell foreground guess 3
    # TODO: check distribution of 31
    
    # create segmentation for WBC detection based on hue and saturation
    sat_min=max(np.sort(cent_2[:,0])[-4],30)
    #mask=np.logical_and(label_mask_resize>1,np.logical_and(hsv_resize[:,:,0]>diag.h_min_wbc,hsv_resize[:,:,0]<diag.h_max_wbc))
    mask=np.logical_and(label_mask_resize>1,hsv_resize[:,:,1]>sat_min)
    if vis_diag:
        imtools.overlayImage(hsv_resize,mask>0,\
        (0,1,0),1,vis_diag=vis_diag,fig='wbc_mask')   
    
    cent_3, label_3 = segmentations.segment_hsv(hsv_resize, mask=mask,\
                                                    cut_channel=1, chs=(0,0,1),\
                                                    n_clusters=4, vis_diag=vis_diag)   

# creating masks for labels        
    n=np.zeros(cent_3.shape[0])
    mask_tmps=[]
    for i, c in enumerate(cent_3):
        mask_tmp=label_3==i
        mask_tmp=morphology.binary_opening(mask_tmp,morphology.disk(1))
        #mask_tmp=morphology.binary_closing(mask_tmp,morphology.disk(1))
        mask_tmps.append(mask_tmp)
        n[i]=(mask_tmps[i]).sum()
#        
    ind_maxn=np.argmax(n)
    # order by saturation - throw away largest area
    # max sat has to be the largest
    ind_sat=np.argsort(cent_3[:,2])
   
    if ind_sat[-1]==ind_maxn:
        mask_pot=mask_tmps[ind_sat[-2]]
    else:
        mask_pot=mask_tmps[ind_sat[-1]]
    im_pot=imtools.overlayImage(im_resize,mask_pot,(0,1,0),1,vis_diag=vis_diag,fig='pot')
   
    cc,num=morphology.label(mask_pot,connectivity=1,return_num=True,background=0)    
    regions = measure.regionprops(cc.astype('int64'))
    
    mask_wbc_small_1=np.zeros(label_3.shape,'uint8') # NE AND EO
    mask_wbc_small_2=np.zeros(label_3.shape,'uint8') # EO AND MONO
    mask_wbc_small_3=np.zeros(label_3.shape,'uint8') # MONO and LYMPHO
    mask_wbc_small_0=np.zeros(label_3.shape,'uint8') # NE 

    
    # MONO, LYMPHO and fuzzy EO
    area=np.zeros(len(regions))
    for i,r in enumerate(regions):
        area[i]=r.area
        if (r.area>np.power((1.1*scale*param.rbcR),2)*np.pi) and\
           (r.area/r.convex_area>0.7):
            mask_wbc_small_1[cc==r.label]=255
            if (r.euler_number==1) and\
                (r.area/r.convex_area>0.8):
                mask_wbc_small_2[cc==r.label]=255
                if (r.area/r.convex_area>0.9) and\
                  (r.eccentricity<0.7):
                      mask_wbc_small_3[cc==r.label]=255
                          
#    area_sorted=np.argsort(area)
#    r=regions[area_sorted[-2]]
#    
    ## Hunting for NE
    if ind_sat[-2]==ind_maxn:
        mask_pot_2=np.logical_or(mask_tmps[ind_sat[-3]],mask_tmps[ind_sat[-1]])
    else:
        mask_pot_2=np.logical_or(mask_tmps[ind_sat[-2]],mask_tmps[ind_sat[-1]])
    im_pot=imtools.overlayImage(im_resize,mask_pot_2,(0,1,0),1,vis_diag=vis_diag,fig='pot')
    
    cc,num=morphology.label(mask_pot_2,connectivity=1,return_num=True,background=0)
    
    regions = measure.regionprops(cc.astype('int64'))
        
    area=np.zeros(len(regions))
    for i,r in enumerate(regions):
        area[i]=r.area
        if (r.area>np.power((1.25*scale*param.rbcR),2)*np.pi) and\
           (r.area/r.convex_area>0.5):
            if (cc[mask_pot>0]==r.label).sum()>r.convex_area*0.25:
                mask_wbc_small_0[cc==r.label]=255
    
#    area_sorted=np.argsort(area)
#    r=regions[area_sorted[-1]]                        
# 
    im_wbc=imtools.overlayImage(im_resize,mask_wbc_small_0>0,(0,1,1),1,vis_diag=False,fig='wbc')
    im_wbc=imtools.overlayImage(im_wbc,mask_wbc_small_1>0,(0,1,0),1,vis_diag=False,fig='wbc')
    im_wbc=imtools.overlayImage(im_wbc,mask_wbc_small_2>0,(1,1,0),1,vis_diag=False,fig='wbc')
    im_wbc=imtools.overlayImage(im_wbc,mask_wbc_small_3>0,(1,0,0),1,vis_diag=vis_diag,fig='wbc')
    
    label_4=-np.ones(hsv_resize.shape[0:2])
    for i, ind in enumerate(ind_sat):
        if not((i==label_3.max()) and (ind==ind_maxn)):
            label_4[label_3==ind]=i
        else:
            print('undersaturated wbc') # wbc is not at highest saturation
            # TODO: add diagnostics
 
    l=imtools.normalize(label_4,vis_diag=vis_diag,fig='labels')

    diag.saveDiagImage(im_wbc,'wbc',savedir=diag_dir)
    diag.saveDiagImage(l,'labels',savedir=diag_dir)
    
    shapelist=[]

    return shapelist
