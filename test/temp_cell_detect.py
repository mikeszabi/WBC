# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:46:42 2017

@author: SzMike
"""
import __init__

from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
import os
import warnings
import skimage.io as io
import numpy as np;
from skimage.transform import resize
from skimage import filters
import glob


from skimage import measure
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

# %matplotlib qt5
 
import cfg
import imtools
import diagnostics
import segmentations
import cell_morphology
import annotations

%matplotlib qt5

image_dir=r'e:\CELLDATA\Slides\1106_kezi_diapH_5_7_12'
#image_dir=r'e:\WBC\data\Test\WBC Types\Problem'

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)


image_file=image_list_indir[0]

vis_diag=False

for image_file in image_list_indir:

    print(image_file)
    
    im = io.imread(image_file) # read uint8 image
   
    diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
    
    output_dir=diag.param.getOutDir(dir_name=os.path.join('output'))
    diag_dir=diag.param.getOutDir(dir_name=os.path.join('diag'))

            
# SMOOTHING
#im_smooth=imtools.smooth3ch(im,r=5)
 
    """
    Foreground masks
    """  
# SMOOTHING
#hsv_smooth=imtools.smooth3ch(diag.hsv_corrected,r=5)         

    
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,diag.param.middle_size,interpolation = 0)
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,diag.param.middle_size,interpolation = 0)
 
# create foreground mask using previously set init centers
    clust_centers_0, label_0 = segmentations.segment_hsv(hsv_resize, init_centers=diag.cent_init,\
                                                         chs=(1,1,2),\
                                                         n_clusters=5,\
                                                         vis_diag=vis_diag)   
    label_fg_bg=cell_morphology.rbc_labels(im,clust_centers_0,label_0)

    """
    WBC nucleus masks
    """
# create segmentation for WBC detection based on hue and saturation
    label_wbc=np.logical_and(np.logical_and(hsv_resize[:,:,0]>diag.param.wbc_range_in_hue[0]*255,\
                                            hsv_resize[:,:,0]<diag.param.wbc_range_in_hue[1]*255),\
                                            hsv_resize[:,:,1]>diag.sat_q95)
    
# TODO: learn wbc range from mask_sat hue distribution
#    im_wbc=imtools.overlayImage(im_resize,mask_sat>0,(0,1,1),0.5,vis_diag=vis_diag,fig='sat')    
#     
#    clust_centers_1, label_1 = segmentations.segment_hsv(hsv_resize, mask=mask_sat,\
#                                                    cut_channel=1, chs=(0,0,0),\
#                                                    n_clusters=2,\
#                                                    vis_diag=vis_diag) 
#    # TODO: remove artifact
#    # find cluster with highest saturation
#
#    clust_hue=clust_centers_1[:,0]
#    
#    clust_sat=np.zeros(len(clust_hue))    
#    label_wbc=np.zeros(label_1.shape)
#    
#    for i in range(clust_hue.shape[0]):
#        hist_hsv=imtools.colorHist(hsv_resize,mask=label_1==i)
#        cumh_hsv, siqr_hsv = diag.semi_IQR(hist_hsv) # Semi-Interquartile Range
#        clust_sat[i]=np.argwhere(cumh_hsv[1]>0.95)[0,0]
#    for i in range(clust_hue.shape[0]):
#        if clust_sat[i]>clust_sat.max()*0.8:       
## TODO: use component size - region props instead
#            label_wbc[label_1==i]=1
##        if clust_hue[i]>diag.param.wbc_range_in_hue[0]*255 and\
##            clust_hue[i]<diag.param.wbc_range_in_hue[1]*255:
##            mask_nuc_2[label_1==i]=1    
## TODO: add clust_hue to diagnostics
    im_wbc=imtools.overlayImage(im_resize,label_wbc>0,(0,1,1),0.5,vis_diag=vis_diag,fig='sat')    

    """
    RBC detection
    """
    label_fg_bg[label_wbc>0]=2
    mask_fg_clear=cell_morphology.rbc_mask_morphology(im_resize,label_fg_bg,diag.param,scale=scale,\
                                                      label_tsh=2,vis_diag=vis_diag,fig='31')    
#      
    markers_rbc=cell_morphology.rbc_markers_from_mask(mask_fg_clear,diag.param,scale=scale)
    segmentation.clear_border(markers_rbc,buffer_size=int(50*scale),in_place=True)
   
#    markers_rbc, rbcR=cell_morphology.blob_markers(label_fg_bg>30,diag.param,scale=scale,fill_tsh=0.85,vis_diag=vis_diag,fig='31')
#    segmentation.clear_border(markers_rbc,buffer_size=diag.param.middle_border,in_place=True)

#    diag.param.rbcR=rbcR
#    diag.measures['rbcR']=rbcR
                 
# TODO: connected component analysis - check if n_RBC can be deduced from component size
# TODO: detailed analysis of RBC counts and sizes

    """
    WBC nucleus detection
    """
    
    markers_wbc_nuc=cell_morphology.wbc_markers(label_wbc>0,diag.param,scale=scale,fill_tsh=0.33,vis_diag=vis_diag,fig='wbc_nuc')
    #segmentation.clear_border(markers_wbc_nuc,buffer_size=diag.param.middle_border,in_place=True)
  
    im_wbc=imtools.overlayImage(im_resize,markers_wbc_nuc,(0,1,0),1,vis_diag=vis_diag,fig='wbc')    
    im_detect=imtools.overlayImage(im_wbc,markers_rbc>0,(1,0,0),1,vis_diag=vis_diag,fig='detections')
    
    diag.saveDiagImage(im_detect,'detections',savedir=diag_dir)