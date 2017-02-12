# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""

import os
import glob
import warnings
import skimage.io as io
import numpy as np;
from skimage.transform import resize
from skimage import morphology
from skimage import feature
from skimage import measure
import matplotlib.pyplot as plt

import _init_path
import cfg
import imtools
import diagnostics
import segmentations
import annotations


plt.close('all')
#%matplotlib qt5
 
##
param=cfg.param()
vis_diag=True

imDirs=os.listdir(param.getTestImageDirs(''))
print(imDirs)
i_imDirs=5
save_dir=param.getSaveDir(imDirs[i_imDirs])
image_dir=param.getTestImageDirs(imDirs[i_imDirs])

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

print(image_list_indir)

for image_file in image_list_indir:
    # reading image
    
#if __name__ == '__main__':
    print(image_file)
    im = io.imread(image_file) # read uint8 image
    
    if vis_diag:
        fo=plt.figure('original image')
        axo=fo.add_subplot(111)
        axo.imshow(im)              
                  
    # diagnose image, create overexpo mask and correct for inhomogen illumination
    diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
    diag.writeDiagnostics(save_dir)
    
    
    """
    Foreground and wbc segmentation
    """                   
    #cent_2, label_mask_2 = segmentations.segment_fg_bg_onechannel_binary(im_corrected, mask_over, diag.measures['ch_maxvar'], vis_diag=True)   
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,512,interpolation = 0)
    
    # create foreground mask using previously set init centers
    cent_2, label_mask_2_resize = segmentations.segment_fg_bg_sv_kmeans4(hsv_resize, diag.cent_init, vis_diag=vis_diag)   
    
    # create segmentation for RBC detection based on hs
    sat_min=np.sort(cent_2[:,0])[-4]/2
    mask=np.logical_and(np.logical_not(label_mask_2_resize==1),hsv_resize[:,:,1]>sat_min)
    cent_3, label_mask_3_resize = segmentations.segment_cell_hs_kmeans3(hsv_resize, mask=mask, cut_channel=1, vis_diag=vis_diag)   
    
    # create segmentation for WBC detection based on hue
    sat_min=np.sort(cent_2[:,0])[-1]*1.2
    mask=np.logical_and(label_mask_3_resize==1,hsv_resize[:,:,1]>sat_min)
    cent_4, label_mask_4_resize = segmentations.segment_wbc_hue(hsv_resize, cut_channel=1, mask=mask, vis_diag=False)   
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label_mask_3 = img_as_ubyte(resize(label_mask_3_resize,(im.shape[0],im.shape[1]), order = 0))
        label_mask_4 = img_as_ubyte(resize(label_mask_4_resize,(im.shape[0],im.shape[1]), order = 0))
    
    """
    Creating Clear RBC mask
    """
    
    # create foreground mask - morphology: fill holes and opening
    #mask_fg_sure=np.logical_or(label_mask_3==1,label_mask_3==3)
    mask_fg_sure=label_mask_3==3
    imtools.maskOverlay(im,255*mask_fg_sure,0.5,vis_diag=vis_diag,fig='mask_fg_sure')
    
    # remove holes from foreground mask
    mask_fg_sure_filled=morphology.remove_small_holes(mask_fg_sure, 
                                                      min_size=param.rbcR*param.rbcR*np.pi, 
                                                      connectivity=1)
    
    # opening
    mask_fg_clear=morphology.binary_opening(mask_fg_sure_filled,morphology.disk(param.rbcR/4)).astype('uint8')
    imtools.maskOverlay(im,255*(mask_fg_clear>0),0.5,vis_diag=vis_diag,fig='mask_fg_sure_clear')
    
    """
    Find RBC markers - using dtf and local maximas
    """
    
    # use dtf to find markers for watershed 
    skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
    dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
    # watershed seeds
    local_maxi = feature.peak_local_max(dtf, indices=False, 
                                        threshold_abs=0.25*param.rbcR,
                                        footprint=np.ones((int(1.25*param.rbcR), int(1.25*param.rbcR))), 
                                        labels=mask_fg_clear.copy())
    markers = measure.label(local_maxi)
    im_markers=imtools.maskOverlay(im,255*morphology.binary_dilation((markers>0),morphology.disk(3)),0.6,ch=0,vis_diag=True,fig='markers') 
     
    #TODO: count markers
    
    
    """
    Save shapes
    """
    #skimage.measure.regionprops
    
    #cnts = measure.find_contours(mask_fg_clear, 0.5)
    #regions = measure.regionprops(labels_ws)
    #http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    
    #if vis_diag:
    #    fc=plt.figure('contours')
    #    axc=fc.add_subplot(111)
    #    axc.imshow(im)   
    #    for n, contour in enumerate(cnts):
    #        axc.plot(contour[:,1], contour[:, 0], linewidth=2)
    #        axc.text(np.mean(contour[:,1]), np.mean(contour[:, 0]),str(n), bbox=dict(facecolor='white', alpha=0.5))
    #    
    
    #for iBlob in range(len(cnts)):
    #    # TODO: check if not close to image border (mean_contour)
    #    # TODO: check if saturation histogram
    #    contour=cnts[iBlob]
    #    mask_blob=np.zeros((im.shape[0],im.shape[1]),'uint8')
    #    #contour=cnts[iBlob]
    #    rr,cc=skimage.draw.polygon(contour[:,0], contour[:, 1], shape=None)
    #    if rr.size>100:
    #        mask_blob[rr,cc]=255
    #        reg = measure.regionprops(mask_blob)
    #        r=reg[0]
    #        nEstimated=r.area/rbcSize
    #       #print(r.convex_area/r.area)
    #       #print(nEstimated)
    #        nMarkers=(local_maxi[mask_blob>0]>0).sum()
    #       #print(nMarkers)
    ##       
    """
    WBC
    """
    # create wbc mask
    
    mask_wbc_sure=label_mask_4==1
    imtools.maskOverlay(im,255*mask_wbc_sure,0.5,vis_diag=vis_diag,fig='mask_wbc_sure')
    
    # opening
    mask_wbc_clear=morphology.binary_opening(mask_wbc_sure,morphology.disk(param.rbcR/3)).astype('uint8')
    im_detect=imtools.maskOverlay(im_markers,255*mask_wbc_clear,0.5,ch=2,vis_diag=True,fig='mask_wbc')
    
    
    """
    SAVE RESULTS
    """
    
    diag.saveDiagImage(im_detect,'_detect',savedir=save_dir)
    
       
    """
    Create SHAPES and store them
    """
    # ToDo: merge groups
    #shapelist=[]
    #for c in cnts:
    #     one_shape=('RBC','general',c,'None','None')
    #     shapelist.append(one_shape)
    #
    #head, tail=os.path.split(image_file)
    #xml_file=os.path.join(save_dir,tail.replace('.bmp',''))
    #tmp = annotations.AnnotationWriter(head,xml_file, (im.shape[0],im.shape[1]))
    #tmp.addShapes(shapelist)
    #tmp.save()
    #
    #imtools.plotShapes(im,shapelist)
    
