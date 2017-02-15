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
from skimage import feature
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

    # SET THE PARAMETERS
    param=cfg.param()
    output_dir=param.getOutDir('output')
    diag_dir=param.getOutDir('diag')
    vis_diag=False
    
    # READ THE IMAGE
    im = io.imread(image_file) # read uint8 image
    # TODO: check image
                  
    # diagnose image, create overexpo mask and correct for inhomogen illumination
    diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
    diag.writeDiagnostics(diag_dir)   
    
    """
    Foreground and wbc segmentation
    """                   
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,512,interpolation = 2)
    label_mask_resize=np.zeros(hsv_resize.shape[0:2]).astype('uint8')

    rgb_range=((330/360*255,30/360*255), (75/360*255,135/360*255), (180/360*255,270/360*255))


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        # create foreground mask using previously set init centers
        cent_2, label_2 = segmentations.segment_fg_bg_sv_kmeans(hsv_resize, diag.cent_init, n_clusters=4, vis_diag=vis_diag)   
        # adding meaningful labels
        ind_sat=np.argsort(cent_2[:,0])
        ind_val=np.argsort(cent_2[:,1])
        label_mask_resize[label_2==ind_val[-1]]=1 # sure background
        label_mask_resize[label_2==ind_sat[-3]]=2 # unsure region
        label_mask_resize[label_2==ind_sat[-1]]=31 # sure cell foreground guess 1 
        if cent_2[ind_sat[-2],0]/cent_2[ind_sat[-3],0]>cent_2[ind_sat[-1],0]/cent_2[ind_sat[-2],0]:
                 label_mask_resize[label_2==ind_sat[-2]]=32 # sure cell foreground guess 2
    
        # create segmentation for WBC detection based on hue and saturation
        sat_min=np.sort(cent_2[:,0])[-3]
        mask=np.logical_and(label_mask_resize>0,hsv_resize[:,:,1]>sat_min)
        mask=np.logical_and(hsv_resize[:,:,1]>sat_min,np.logical_and(hsv_resize[:,:,0]>rgb_range[2][0]-10,hsv_resize[:,:,0]<rgb_range[2][1]+10))
        cent_3, label_3 = segmentations.segment_cell_hs_kmeans(hsv_resize, mask=mask, cut_channel=1, n_clusters=3, vis_diag=vis_diag)   
        # adding meaningful labels
        ind_sat=np.argsort(cent_3[:,1])
        label_mask_resize[label_3==ind_sat[-1]]=4 # sure wbc
        label_mask_resize[label_3==ind_sat[0]]=4 # sure wbc    
             
        label_mask = img_as_ubyte(resize(label_mask_resize,diag.image_shape, order = 0))
        
    """
    Creating Clear RBC mask
    """
   
    mask_fg_clear=cell_morphology.rbc_mask_morphology(im,label_mask,param,vis_diag=vis_diag,fig='31')
    
    
    """
    Find RBC markers - using dtf and local maximas
    """
    
    # use dtf to find markers for watershed 
    skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
    dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
    # watershed seeds
    # TODO - add parameters to cfg
    local_maxi = feature.peak_local_max(dtf, indices=False, 
                                        threshold_abs=0.25*param.rbcR,
                                        footprint=np.ones((int(1.25*param.rbcR), int(1.25*param.rbcR))), 
                                        labels=mask_fg_clear.copy())
    markers, n_RBC = measure.label(local_maxi,return_num=True)
    segmentation.clear_border(markers,buffer_size=50,in_place=True)
    #TODO: count markers
    
    """
    WBC
    """
    # create wbc mask
    
    mask_wbc_nucleus=label_mask==4
    #imtools.maskOverlay(im,255*mask_wbc_nucleus,0.5,vis_diag=vis_diag,fig='mask_wbc_nucleus')
#    mask_wbc_nucleus=morphology.binary_closing(mask_wbc_nucleus,morphology.disk(param.rbcR/4)).astype('uint8')
#    mask_wbc_nucleus=morphology.binary_opening(mask_wbc_nucleus,morphology.disk(param.rbcR)).astype('uint8')
#       
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,256,interpolation = 0)
    mask_wbc_nucleus_small, scale=imtools.imRescaleMaxDim(mask_wbc_nucleus,256,interpolation = 0)
    mask_bckg_small, scale=imtools.imRescaleMaxDim(label_mask==1,256,interpolation = 0)

    circ=morphology.disk(scale*param.rbcR)
    circ_response = 255*img_as_float(filters.rank.mean(mask_wbc_nucleus_small>0, selem=circ, mask=(label_mask>1)))

#    segments_small = segmentation.slic(im_resize, n_segments=markers.max()*2, compactness=10).astype('float64')
#    #imtools.overlayImage(im_resize,segmentation.find_boundaries(segments_small),(0,1,0),1,vis_diag=vis_diag,fig='wbc')
#    
#    mask_wbc_small=np.zeros(im_resize.shape[0:2]).astype('uint8')
#    seg_2=segments_small[mask_wbc_nucleus_small>0]
#    for i in np.unique(seg_2):
#        if (seg_2==i).sum()>0:
#            print(i)
#            mask_wbc_small[segments_small==i]=1
#            mask_wbc_small[mask_bckg_small>0]=0    
#    mask_wbc_small=morphology.binary_opening(mask_wbc_small,morphology.disk(5)).astype('uint8')
#    mask_wbc_small=morphology.binary_dilation(mask_wbc_small,morphology.disk(2)).astype('uint8')
#    
    mask_wbc_small=circ_response>128
    
    
    imtools.overlayImage(im_resize,mask_wbc_small>0,(0,1,0),1,vis_diag=vis_diag,fig='wbc')
      
#    
#    circ=morphology.disk(param.rbcR/3)
#    circ_response = 255*img_as_float(filters.rank.mean(mask_wbc_nucleus>0, selem=circ, mask=(label_mask>1)))
#    circ_response.flat[(circ_response>0).flatten()]+=np.random.random(((circ_response>0).sum()))
#    local_maxi = feature.peak_local_max(circ_response, indices=False, 
#                                        threshold_abs=30,
#                                        footprint=np.ones((int(2*param.rbcR), int(2*param.rbcR))), 
#                                        labels=mask_wbc_nucleus.copy())
#
#    markers_WBC, n_WBC = measure.label(local_maxi,return_num=True)
#    mask_wbc=morphology.binary_dilation(markers_WBC>0,morphology.disk(2*param.rbcR))
#        
    markers[mask_wbc>0]=0

    wbc_bound=segmentation.find_boundaries(mask_wbc>0)
    im_wbc=imtools.overlayImage(im,morphology.binary_dilation(wbc_bound>0,morphology.disk(5)),\
            (1,1,1),1,vis_diag=False)
    im_detect=imtools.overlayImage(im_wbc,morphology.binary_dilation(markers>0,morphology.disk(5)),\
            (1,0,0),1,vis_diag=vis_diag,fig='detections')


    """
    SAVE IMAGE RESULTS
    """
    
    diag.saveDiagImage(im_detect,'_detect',savedir=diag_dir)
    

    """
    Save shapes
    """
    #skimage.measure.regionprops
    
    cnts_RBC = measure.find_contours(markers>0, 0.5)
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
    if vis_diag:
        imtools.plotShapes(im,shapelist)
    
    return
