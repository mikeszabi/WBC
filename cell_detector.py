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
    diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
    diag.writeDiagnostics(diag_dir)   
    
    """
    Foreground and wbc segmentation
    """                   
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,param.middle_size,interpolation = 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        # create foreground mask using previously set init centers
        cent_2, label_2 = segmentations.segment_hsv(hsv_resize, init_centers=diag.cent_init, chs=(1,1,2), n_clusters=4, vis_diag=vis_diag)   
        cent_dist=segmentations.center_diff_matrix(cent_2,metric='euclidean')
        
        # adding meaningful labels
        ind_sat=np.argsort(cent_2[:,0])
        ind_val=np.argsort(cent_2[:,2])
        
        label_mask_resize=np.zeros(hsv_resize.shape[0:2]).astype('uint8')
        label_mask_resize[label_2==ind_val[-1]]=1 # sure background
        label_mask_resize[label_2==ind_sat[-3]]=2 # unsure region
        label_mask_resize[label_2==ind_sat[-1]]=31 # sure cell foreground guess 1 
        if cent_dist[ind_sat[-1],ind_sat[-2]]<cent_dist[ind_sat[-2],ind_sat[-4]]:
           label_mask_resize[label_2==ind_sat[-2]]=32 # sure cell foreground guess 2
           if cent_dist[ind_sat[-2],ind_sat[-3]]<cent_dist[ind_sat[-2],ind_sat[-4]]:                 
               label_mask_resize[label_2==ind_sat[-3]]=33 # sure cell foreground guess 3
   
        # create segmentation for WBC detection based on hue and saturation
        #sat_min=max(np.sort(cent_2[:,0])[-3],30)
        mask=np.logical_and(label_mask_resize>2,np.logical_and(hsv_resize[:,:,0]>diag.h_min_wbc,hsv_resize[:,:,0]<diag.h_max_wbc))
        if vis_diag:
            imtools.overlayImage(hsv_resize,mask>0,\
            (0,1,0),1,vis_diag=vis_diag,fig='wbc_mask')   
        
        #cent_3, label_3 = segmentations.segment_hsv(hsv_resize, mask=mask, cut_channel=1, chs=(0,1,2), n_clusters=6, vis_diag=vis_diag)   
        # adding meaningful labels
        #ind_sat=np.argsort(cent_3[:,1])

        label_mask_resize_wbc=np.zeros(hsv_resize.shape[0:2]).astype('uint8')       
        label_mask_resize_wbc[mask]=30
        
#        for i, c in enumerate(cent_3):
#            if (abs(c[0]-cent_3[ind_sat[-1],0]))<(diag.h_max_wbc-diag.h_min_wbc)/10:
#                if i==ind_sat[-1]:
#                     print(i)
#                     label_mask_resize_wbc[label_3==i]=30 # potential wbc
#                elif (cent_3[ind_sat[-1],2]/c[2]>2):   
#                    label_mask_resize_wbc[label_3==i]=30 # potential wbc
#   
             
        label_mask = img_as_ubyte(resize(label_mask_resize,diag.image_shape, order = 0))
        label_mask_wbc = img_as_ubyte(resize(label_mask_resize_wbc,diag.image_shape, order = 0))
        label_mask[label_mask_wbc==30]=30
    """
    Creating Clear RBC mask with morphology
    """
   
    mask_fg_clear=cell_morphology.rbc_mask_morphology(im,label_mask,label_tsh=3,vis_diag=vis_diag,fig='31')
    
    
    """
    Find RBC markers - using dtf and local maximas
    """
    
    markers_rbc=cell_morphology.rbc_markers_from_mask(mask_fg_clear)
    if vis_diag:
        imtools.overlayImage(im,morphology.binary_dilation(markers_rbc>0,morphology.disk(5))>0,\
            (1,0,0),0.5,vis_diag=vis_diag,fig='rbc_markers')   
    
    """
    WBC
    """
    # create wbc mask
    
    mask_wbc=label_mask_wbc==30
    
    mask_wbc_small, scale=imtools.imRescaleMaxDim(mask_wbc,param.middle_size,interpolation = 0)
    mask_wbc_small=morphology.binary_opening(mask_wbc_small,morphology.disk(scale*param.rbcR))

#    circ=morphology.disk(scale*1.5*param.rbcR)
#    circ_response = 255*img_as_float(filters.rank.mean(mask_wbc_small>0, selem=circ))
#    mask_wbc_small=circ_response>255/3 # fill pct

    mask_wbc=resize(mask_wbc_small,diag.image_shape,order=0)        
    
    markers_rbc[morphology.binary_dilation(mask_wbc>0,morphology.disk(0.1*param.rbcR))>0]=0

    wbc_bound=segmentation.find_boundaries(mask_wbc>0)
    im_wbc=imtools.overlayImage(im,morphology.binary_dilation(wbc_bound>0,morphology.disk(3)),\
            (1,1,1),1,vis_diag=False)
    im_detect=imtools.overlayImage(im_wbc,morphology.binary_dilation(markers_rbc>0,morphology.disk(5)),\
            (1,0,0),1,vis_diag=False,fig='detections')
    border=np.zeros(diag.image_shape).astype('uint8')
    border[0:50,:]=1
    border[-51:-1,:]=1     
    border[:,0:50]=1
    border[:,-51:-1]=1    
    im_detect=imtools.overlayImage(im_detect,border>0,\
            (1,1,0),0.2,vis_diag=vis_diag,fig='detections')      
    """
    SAVE IMAGE RESULTS
    """
    
    diag.saveDiagImage(im_detect,'_detect',savedir=diag_dir)
    

    """
    Save shapes
    """
    #skimage.measure.regionprops
    
    cnts_RBC = measure.find_contours(markers_rbc>0, 0.5)
    cnts_WBC = measure.find_contours(mask_wbc>0, 0.5)

    
    if vis_diag:
        fc=plt.figure('contours')
        axc=fc.add_subplot(111)
        axc.imshow(im)   
        for n, contour in enumerate(cnts_RBC):
            axc.plot(contour[:,1], contour[:, 0], linewidth=3, color='r')
 #           axc.text(np.mean(contour[:,1]), np.mean(contour[:, 0]),str(n), bbox=dict(facecolor='white', alpha=0.5))
        for n, contour in enumerate(cnts_WBC):
            axc.plot(contour[:,1], contour[:, 0], linewidth=3, color='b')
    

    # ToDo: merge groups
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
    
    
    return shapelist
