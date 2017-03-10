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
from skimage import segmentation

# %matplotlib qt5
 
import __init__
import imtools
import diagnostics
import segmentations
import cell_morphology
import annotations


def batch_cell_detector(image_dir,save_diag=False,out_dir=''): 
    
    if not os.path.exists(image_dir):
        print('directory does not exists')
        return
    
    included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
    image_list_indir = []
    for ext in included_extenstions:
        image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))
        
    for image_file in image_list_indir:    
        print(image_file)
        cell_detector(image_file,save_diag,out_dir=out_dir)

def cell_detector(image_file,save_diag=False,out_dir=''): 
    
    vis_diag=False
    
# OPEN THE image to be processed
    try:
        im = io.imread(image_file) # read uint8 image
    except Exception:
        print(image_file+' does not exist')
        return []
    if im.ndim!=3:
        print('not color image')
        return []    
   
# SET THE PARAMETERS and DO DIAGNOSTICS
# diagnose image, create overexpo mask and correct for inhomogen illumination
    diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
    
    output_dir=diag.param.getOutDir(dir_name=os.path.join('output',out_dir))
    diag_dir=diag.param.getOutDir(dir_name=os.path.join('diag',out_dir))

            
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
# TODO: add clust_hue to diagnostics
# TODO: learn wbc range from mask_sat hue distribution

    """
    RBC detection
    """
    label_fg_bg[label_wbc>0]=2
    mask_fg_clear=cell_morphology.rbc_mask_morphology(im_resize,label_fg_bg,diag.param,scale=scale,\
                                                      label_tsh=2,vis_diag=vis_diag,fig='31')
#      
    markers_rbc=cell_morphology.rbc_markers_from_mask(mask_fg_clear,diag.param,scale=scale)
    segmentation.clear_border(markers_rbc,buffer_size=int(50*scale),in_place=True)
   
# TODO: connected component analysis - check if n_RBC can be deduced from component size
# TODO: detailed analysis of RBC counts and sizes

    """
    WBC nucleus detection
    """
    
    markers_wbc_nuc=cell_morphology.wbc_markers(label_wbc>0,diag.param,scale=scale,\
                                                fill_tsh=0.33,vis_diag=vis_diag,fig='wbc_nuc')
    segmentation.clear_border(markers_wbc_nuc,buffer_size=diag.param.middle_border,in_place=True)
  
    """
    CHECK ERRORS
    """
    diag.checks()
    if len(diag.error_list)>0:
        print(image_file+' is of wrong quality')
        return []
    
    """
    CREATE shapes
    """
    cnts_RBC = measure.find_contours(markers_rbc>0, 0.5)
    cnts_WBC = measure.find_contours(markers_wbc_nuc>0, 0.5)
    
    shapelist=[]
    for c in cnts_RBC:
         c=np.reshape(np.average(c,axis=0),(1,2))
         pts=[]
         for yx in c:
             pts.append((yx[1]/scale,yx[0]/scale))
         one_shape=('RBC','point',pts,'None','None')
         shapelist.append(one_shape)
    for c in cnts_WBC:
         c=np.reshape(np.average(c,axis=0),(1,2))
         pts=[]
         for yx in c:
             pts.append((yx[1]/scale,yx[0]/scale))
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
        im_wbc=imtools.overlayImage(im_resize,markers_wbc_nuc>0,(0,1,0),1,vis_diag=vis_diag,fig='wbc')    
        im_detect=imtools.overlayImage(im_wbc,markers_rbc>0,(1,0,0),1,vis_diag=False,fig='detections')
        border=np.zeros(im_resize.shape[0:2]).astype('uint8')
        border[0:diag.param.middle_border,:]=1
        border[-diag.param.middle_border-1:-1,:]=1     
        border[:,0:diag.param.middle_border]=1
        border[:,-diag.param.middle_border-1:-1]=1    
        im_detect=imtools.overlayImage(im_detect,border>0,\
                (1,1,0),0.2,vis_diag=vis_diag,fig='detections')       
        im_detect,scale=imtools.imRescaleMaxDim(im_detect,diag.param.middle_size,interpolation = 1)
        
        diag.saveDiagImage(im_detect,'detections',savedir=diag_dir)
        diag.writeDiagnostics(diag_dir)   


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
    parser.add_argument('-o', action='store', dest='o', type=str, required=False,
                    default='')

# Parse the arguments
    inargs = parser.parse_args()
    path_str = os.path.abspath(inargs.i)
    
    if inargs.b is None:
        print('Single image process')
        cell_detector(path_str,save_diag=inargs.s==inargs.s,out_dir=inargs.o)
    else:
        print('Batch execution')
        batch_cell_detector(path_str,save_diag=inargs.s==inargs.s,out_dir=inargs.o)    
    sys.exit(1)
  