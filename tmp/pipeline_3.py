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
from skimage import segmentation
from skimage import color
from skimage import img_as_ubyte, img_as_float
import skimage.morphology 
import matplotlib.pyplot as plt
from scipy import stats

import _init_path
import cfg
import imtools
import diagnostics
import segmentations
import annotations
import random

plt.close('all')
%matplotlib qt5
 
##
param=cfg.param()
vis_diag=False

imDirs=os.listdir(param.getTestImageDirs(''))
print(imDirs)
i_imDirs=5
image_dir=param.getTestImageDirs(imDirs[i_imDirs])
print(glob.glob(os.path.join(image_dir,'*.bmp')))
image_file=os.path.join(image_dir,'45_MO.jpg')
save_dir=param.getSaveDir(imDirs[i_imDirs])

# reading image
im = io.imread(image_file) # read uint8 image

if vis_diag:
    fo=plt.figure('original image')
    axo=fo.add_subplot(111)
    axo.imshow(im)              
              
# diagnose image
diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
diag.writeDiagnostics(save_dir)
"""
inhomogen illumination correction
"""

# create sure background with n=4 kmeans, on sv channel
cent, label_mask_small = segmentations.segment_fg_bg_sv_kmeans4(diag.csp_small, 'k-means++', vis_diag=vis_diag)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    label_mask = img_as_ubyte(resize(label_mask_small,( im.shape[0],im.shape[1]), order = 0))

mask_bg_sure= np.logical_or(label_mask == 0, label_mask == 1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    csp_corrected= diagnostics.illumination_inhomogenity(diag.csp, mask_bg_sure, vis_diag=vis_diag)
    im_corrected=img_as_ubyte(color.hsv2rgb(csp_corrected))
if vis_diag:
    fo=plt.figure('intensity corrected image')
    axo=fo.add_subplot(111)
    axo.imshow(im_corrected)

# TODO: store inhomogenity measure for illumination (pct)
# TODO: check background distance transform and coverage (area) - should not be too large, too small

"""
Creating edges
"""
# create edges on ch_maxvar
#edge_mag=imtools.getGradientMagnitude(im_corrected[:,:,diag.measures['ch_maxvar']]).astype('float64')
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    tmp=imtools.normalize(img_as_ubyte(edge_mag),vis_diag=vis_diag,fig='edges')
###    
 
    
"""
Foreground and wbc segmentation
"""                   
#cent_2, label_mask_2 = segmentations.segment_fg_bg_onechannel_binary(im_corrected, mask_over, diag.measures['ch_maxvar'], vis_diag=True)   
csp_resize, scale=imtools.imRescaleMaxDim(csp_corrected,512,interpolation = 1)

cent_2, label_mask_2_resize = segmentations.segment_fg_bg_sv_kmeans4(csp_resize, cent, vis_diag=vis_diag)   
sat_min=0 #np.sort(cent_2[:,0])[-4]/2

mask=np.logical_and(np.logical_not(label_mask_2_resize==1),csp_resize[:,:,1]>sat_min)
#mask=np.logical_not(label_mask_2_resize==1)

#imtools.maskOverlay(csp_resize,255*mask,0.5,vis_diag=vis_diag,fig='mask_fg_sure')
cent_3, label_mask_3_resize = segmentations.segment_cell_hs_kmeans3(csp_resize, mask=mask, cut_channel=1, vis_diag=vis_diag)   

sat_min=np.sort(cent_2[:,0])[-1]*1.2

mask=np.logical_and(label_mask_3_resize==1,csp_resize[:,:,1]>sat_min)
cent_4, label_mask_4_resize = segmentations.segment_wbc_hue(csp_resize, cut_channel=1, mask=mask, vis_diag=vis_diag)   

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    label_mask_3 = img_as_ubyte(resize(label_mask_3_resize,(im.shape[0],im.shape[1]), order = 0))
    label_mask_4 = img_as_ubyte(resize(label_mask_4_resize,(im.shape[0],im.shape[1]), order = 0))

"""
Creating Clear FG mask
"""

# create foreground mask
mask_fg_sure=np.logical_or(label_mask_3==1,label_mask_3==3)
imtools.maskOverlay(im,255*mask_fg_sure,0.5,vis_diag=vis_diag,fig='mask_fg_sure')

# remove holes from foreground mask

mask_fg_sure_filled=morphology.remove_small_holes(mask_fg_sure, min_size=param.rbcR*param.rbcR*np.pi, connectivity=4)

# opening
mask_fg_clear=morphology.binary_opening(mask_fg_sure_filled,morphology.disk(param.rbcR/3)).astype('uint8')
imtools.maskOverlay(im,255*(mask_fg_clear>0),0.5,vis_diag=vis_diag,fig='mask_fg_sure_clear')

"""
Find RBC markers - using dtf and local maximas
"""

# use dtf to find markers for watershed 
skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
# watershed seeds
local_maxi = feature.peak_local_max(dtf, indices=False, 
                                    threshold_abs=0.5*param.rbcR,
                                    footprint=np.ones((int(1.25*param.rbcR), int(1.25*param.rbcR))), 
                                    labels=mask_fg_clear.copy())
# Problem - similar maximas are left
# remove noisy maximas
markers = measure.label(local_maxi)
im_markers=imtools.maskOverlay(im,255*morphology.binary_dilation((markers>0),morphology.disk(3)),0.6,ch=0,vis_diag=True,fig='markers') 
 
#TODO: count markers

# watershed on dtf
#labels_ws = morphology.watershed(-dtf, markers, mask=mask_fg_clear)

# edge map for visualization
#bounds=segmentation.find_boundaries(labels_ws,connectivity=3).astype('uint8')*255
#
#im2=imtools.maskOverlay(im,bounds,0.5,ch=1,sbs=False,vis_diag=vis_diag,fig='contours')


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

