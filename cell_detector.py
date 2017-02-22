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
from skimage import img_as_ubyte

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
    output_dir=param.getOutDir(dir_name='output')
    diag_dir=param.getOutDir(dir_name='diag')
    
    # READ THE IMAGE
    im = io.imread(image_file) # read uint8 image
    # TODO: check if image exists
            
    # SMOOTHING
    #im_smooth=imtools.smooth3ch(im,r=5)
     
    # diagnose image, create overexpo mask and correct for inhomogen illumination
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
    diag.writeDiagnostics(diag_dir)   
    
    """
    Foreground masks
    """                   
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,param.middle_size,interpolation = 0)
    #im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,param.middle_size,interpolation = 0)
 
    # create foreground mask using previously set init centers
    clust_centers_0, label_0 = segmentations.segment_hsv(hsv_resize, init_centers=diag.cent_init,\
                                                         chs=(1,1,2),\
                                                         n_clusters=4,\
                                                         vis_diag=vis_diag)   
    label_fg_bg=cell_morphology.rbc_labels(im,clust_centers_0,label_0)

    """
    WBC masks
    """
# create segmentation for WBC detection based on hue and saturation
    sat_min=max(np.sort(clust_centers_0[:,0])[0],30)
    #mask=np.logical_and(label_fg_bg>1,np.logical_and(hsv_resize[:,:,0]>diag.h_min_wbc,hsv_resize[:,:,0]<diag.h_max_wbc))
    mask_not_bckg=np.logical_and(label_fg_bg>1,hsv_resize[:,:,1]>sat_min)
       
    clust_centers_1, label_1 = segmentations.segment_hsv(hsv_resize, mask=mask_not_bckg,\
                                                    cut_channel=1, chs=(0,0,1),\
                                                    n_clusters=4,\
                                                    vis_diag=vis_diag)   

    mask_wbc_pot=cell_morphology.wbc_masks(label_1,clust_centers_1,scale,vis_diag=vis_diag)
    label_wbc=np.zeros(mask_wbc_pot[0].shape,'uint8') 
    # NE,EO; NE,EO; EO,MO; MO,LY

# DETECTION - mainly Neutrophil
    cc,num=morphology.label(mask_wbc_pot[1],connectivity=1,return_num=True,background=0)    
    regions = measure.regionprops(cc.astype('int64'))        
    area=np.zeros(len(regions))
    for i,r in enumerate(regions):
        area[i]=r.area
        if (r.area>np.power((1.25*scale*param.rbcR),2)*np.pi) and\
           (r.area/r.convex_area>0.5):
            if (cc[mask_wbc_pot[0]>0]==r.label).sum()>r.convex_area*0.25:
                label_wbc[cc==r.label]=1

# DETECTION - NE, MONO, LYMPHO and fuzzy EO
    cc,num=morphology.label(mask_wbc_pot[0],connectivity=1,return_num=True,background=0)    
    regions = measure.regionprops(cc.astype('int64'))
    area=np.zeros(len(regions))
    for i,r in enumerate(regions):
        area[i]=r.area
        if r.area>np.power((0.9*scale*param.rbcR),2)*np.pi:
            # Lymphocites are sometimes small
            if (r.area/r.convex_area>0.9) and\
                  (r.eccentricity<0.5):
                      label_wbc[cc==r.label]=4
        if (r.area>np.power((1.2*scale*param.rbcR),2)*np.pi) and\
           (r.area/r.convex_area>0.7):
            label_wbc[cc==r.label]=2
            if (r.euler_number==1) and\
               (r.area/r.convex_area>0.8):
                label_wbc[cc==r.label]=3
                if (r.area/r.convex_area>0.9) and\
                   (r.eccentricity<0.7):
                   label_wbc[cc==r.label]=4

                          
#    area_sorted=np.argsort(area)
#    r=regions[area_sorted[-2]]
#         

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
   
    mask_fg_clear=cell_morphology.rbc_mask_morphology(im,label_fg_bg_orig,label_tsh=3,vis_diag=vis_diag,fig='31')    
    markers_rbc=cell_morphology.rbc_markers_from_mask(mask_fg_clear)

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

    im_wbc=imtools.overlayImage(im,label_wbc_orig==1,(0,1,1),1,vis_diag=False,fig='wbc')
    im_wbc=imtools.overlayImage(im_wbc,label_wbc_orig==2,(1,0,1),1,vis_diag=False,fig='wbc')
    im_wbc=imtools.overlayImage(im_wbc,label_wbc_orig==3,(1,1,0),1,vis_diag=False,fig='wbc')
    im_wbc=imtools.overlayImage(im_wbc,label_wbc_orig==4,(0,0,1),1,vis_diag=vis_diag,fig='wbc')   
    im_detect=imtools.overlayImage(im_wbc,morphology.binary_dilation(markers_rbc>0,morphology.disk(5)),\
            (1,0,0),1,vis_diag=False,fig='detections')
    border=np.zeros(diag.image_shape).astype('uint8')
    border[0:50,:]=1
    border[-51:-1,:]=1     
    border[:,0:50]=1
    border[:,-51:-1]=1    
    im_detect=imtools.overlayImage(im_detect,border>0,\
            (1,1,0),0.2,vis_diag=vis_diag,fig='detections')       

    diag.saveDiagImage(im_detect,'detections',savedir=diag_dir)
    

    return shapelist
