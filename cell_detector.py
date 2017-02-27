# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:21:59 2017

@author: SzMike
"""
import os
import sys
import glob
import warnings
import argparse

import numpy as np;
import skimage.io as io
from skimage.transform import resize
from skimage import morphology
from skimage import measure
from skimage import img_as_ubyte

# %matplotlib qt5
 
import __init__
import imtools
import diagnostics
import segmentations
import cell_morphology
import annotations


def batch_cell_detector(image_dir,save_diag=False): 
    
    if not os.path.exists(image_dir):
        print('directory does not exists')
        return
    
    included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
    image_list_indir = []
    for ext in included_extenstions:
        image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))
        
    for image_file in image_list_indir:    
        print(image_file)
        cell_detector(image_file,save_diag)

def cell_detector(image_file,save_diag=False): 

    # OPEN THE image to be processed
    try:
        im = io.imread(image_file) # read uint8 image
    except Exception:
        print(image_file+' does not exist')
        return []
    if im.ndim!=3:
        print('not color image')
        return []
    
    vis_diag=False
   
    # SET THE PARAMETERS and DO DIAGNOSTICS
    # diagnose image, create overexpo mask and correct for inhomogen illumination
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
    
    output_dir=diag.param.getOutDir(dir_name='output')
    diag_dir=diag.param.getOutDir(dir_name='diag')

    diag.writeDiagnostics(diag_dir)   
            
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
                                                         n_clusters=4,\
                                                         vis_diag=vis_diag)   
    label_fg_bg=cell_morphology.rbc_labels(im,clust_centers_0,label_0)

    """
    WBC nucleus masks
    """
    # create segmentation for WBC detection based on hue and saturation
    mask_sat=hsv_resize[:,:,1]>diag.sat_q90
    
    #mask_wbc=morphology.binary_opening(mask_wbc,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR)))
       
    clust_centers_1, label_1 = segmentations.segment_hsv(hsv_resize, mask=mask_sat,\
                                                    cut_channel=1, chs=(0,0,0),\
                                                    n_clusters=3,\
                                                    vis_diag=vis_diag) 
    # find cluster with highest saturation

    clust_hue=clust_centers_1[:,0]
    
    clust_sat=np.zeros(len(clust_hue))    
    label_wbc=np.zeros(label_1.shape)
    
    for i in range(clust_hue.shape[0]):
        hist_hsv=imtools.colorHist(hsv_resize,mask=label_1==i)
        cumh_hsv, siqr_hsv = diag.semi_IQR(hist_hsv) # Semi-Interquartile Range
        clust_sat[i]=np.argwhere(cumh_hsv[1]>0.99)[0,0]
    for i in range(clust_hue.shape[0]):
        if clust_sat[i]==clust_sat.max():
            mask_temp=label_1==i
            mask_temp=morphology.binary_closing(mask_temp,morphology.disk(np.round(1.5*diag.param.cell_bound_pct*diag.param.rbcR*scale)))           
            mask_temp=morphology.binary_opening(mask_temp,morphology.disk(np.round(2*diag.param.cell_bound_pct*diag.param.rbcR*scale))) 
            mask_temp=morphology.binary_closing(mask_temp,morphology.disk(np.round(1.5*diag.param.rbcR*scale)))                       
# TODO: use rbc size for morphology
# TODO: use component size instead
#            mask_temp=morphology.binary_opening(mask_temp,morphology.disk(int(scale*diag.param.cell_bound_pct*diag.param.rbcR)))            
            label_wbc[mask_temp]=1
#        if clust_hue[i]>diag.param.wbc_range_in_hue[0]*255 and\
#            clust_hue[i]<diag.param.wbc_range_in_hue[1]*255:
#            mask_nuc_2[label_1==i]=1    
   # TODO: add clust_hue to diagnostics
    """
    RESIZE MASKS TO ORIGINAL
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        label_fg_bg_orig = img_as_ubyte(resize(label_fg_bg,diag.image_shape, order = 0))
        label_wbc_orig = img_as_ubyte(resize(label_wbc,diag.image_shape, order = 0))
        label_fg_bg_orig[label_wbc_orig>0]=0
    
    """
    RBC detection
    """
   
    mask_fg_clear=cell_morphology.rbc_mask_morphology(im,label_fg_bg_orig,diag.param,label_tsh=3,vis_diag=vis_diag,fig='31')    
    markers_rbc=cell_morphology.rbc_markers_from_mask(mask_fg_clear,diag.param)

    # TODO: connected component analysis - check if n_RBC can be deduced from component size

    """
    """
    cnts_RBC = measure.find_contours(markers_rbc>0, 0.5)
    cnts_WBC = measure.find_contours(label_wbc_orig>0, 0.5)
    
    shapelist=[]
    for c in cnts_RBC:
         c=np.reshape(np.average(c,axis=0),(1,2))
         pts=[]
         for yx in c:
             pts.append((yx[1],yx[0]))
         one_shape=('RBC','point',pts,'None','None')
         shapelist.append(one_shape)
    for c in cnts_WBC:
         c=np.reshape(np.average(c,axis=0),(1,2))
         pts=[]
         for yx in c:
             pts.append((yx[1],yx[0]))
         one_shape=('WBC','polygon',pts,'None','None')
         shapelist.append(one_shape)
    
    head, tail=os.path.split(image_file)
    xml_file=os.path.join(output_dir,tail.replace('.bmp',''))
    tmp = annotations.AnnotationWriter(head,xml_file, (im.shape[0],im.shape[1]))
    tmp.addShapes(shapelist)
    tmp.save()

    """
    CREATE and SAVE DIAGNOSTICS IMAGES
    """
    if save_diag:
        wbc_mask=imtools.overlayImage(im_resize,mask_sat>0,(1,1,0),0.5,vis_diag=False,fig='wbc_mask')
        wbc_mask=imtools.overlayImage(wbc_mask,label_wbc>0,(0,1,0),1,vis_diag=vis_diag,fig='wbc_mask')
        diag.saveDiagImage(wbc_mask,'wbc_nucleus_mask',savedir=diag_dir)
        
        im_wbc=imtools.overlayImage(im,label_wbc_orig>0,(0,1,0),1,vis_diag=vis_diag,fig='wbc')    
        im_detect=imtools.overlayImage(im_wbc,morphology.binary_dilation(markers_rbc>0,morphology.disk(5)),\
                (1,0,0),1,vis_diag=False,fig='detections')
        border=np.zeros(diag.image_shape).astype('uint8')
        border[0:50,:]=1
        border[-51:-1,:]=1     
        border[:,0:50]=1
        border[:,-51:-1]=1    
        im_detect=imtools.overlayImage(im_detect,border>0,\
                (1,1,0),0.2,vis_diag=vis_diag,fig='detections')       
        im_detect,scale=imtools.imRescaleMaxDim(im_detect,diag.param.middle_size,interpolation = 1)
        diag.saveDiagImage(im_detect,'detections',savedir=diag_dir)
    

    return shapelist

if __name__=='__main__':
    # Initialize argument parse object
    parser = argparse.ArgumentParser()

    # This would be an argument you could pass in from command line
    parser.add_argument('-i', action='store', dest='i', type=str, required=True,
                    default='')
    parser.add_argument('-b', action='store', dest='b', type=str, required=False,
                    default=None)
    parser.add_argument('-s', action='store', dest='s', type=bool, required=False,
                    default=False)

    # Parse the arguments
    inargs = parser.parse_args()
    path_str = os.path.abspath(inargs.i)
    print(path_str)
    
    if inargs.b is None:
        print('Single image process')
        cell_detector(path_str,save_diag=inargs.s)
    else:
        print('Batch execution')
        batch_cell_detector(path_str,save_diag=inargs.s)    
    sys.exit(1)
  