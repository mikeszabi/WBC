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
from skimage.color import rgb2hsv, hsv2rgb
from skimage import img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu
import skimage.morphology 
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import chi2
%matplotlib qt5

import _init_path
import cfg
import imtools
import diagnostics
import segmentations
import annotations
import random
 
##
param=cfg.param()
vis_diag=False

imDirs=os.listdir(param.getImageDirs(''))
print(imDirs)
i_imDirs=1
image_dir=param.getImageDirs(imDirs[i_imDirs])
print(glob.glob(os.path.join(image_dir,'*.bmp')))
image_file=os.path.join(image_dir,'72.bmp')
save_dir=param.getSaveDir(imDirs[i_imDirs])

# reading image
im = io.imread(image_file) # read uint8 image

if vis_diag:
    fo=plt.figure('original image')
    axo=fo.add_subplot(111)
    axo.imshow(im)              
              
# diagnose image
diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
#diag.writeDiagnostics(save_dir)
"""
inhomogen illumination correction
"""

# create sure background with n=4 kmeans, on sv channel
cent, label_mask_small = segmentations.segment_fg_bg_sv_kmeans4(diag.hsv_small, 'k-means++', vis_diag=vis_diag)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    label_mask = img_as_ubyte(resize(label_mask_small,( im.shape[0],im.shape[1]), order = 0))

mask_bg_sure= np.logical_or(label_mask == 0, label_mask == 1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hsv_corrected= diagnostics.illumination_inhomogenity(diag.hsv, mask_bg_sure, vis_diag=vis_diag)
    im_corrected=img_as_ubyte(hsv2rgb(hsv_corrected))
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
edge_mag=imtools.getGradientMagnitude(im_corrected[:,:,diag.measures['ch_maxvar']]).astype('float64')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tmp=imtools.normalize(img_as_ubyte(edge_mag),vis_diag=vis_diag)
##    
##threshold_global_otsu = threshold_otsu(edge_mag)
##edges = edge_mag >= threshold_global_otsu
#edges = feature.canny(hsv_corrected[:,:,diag.measures['ch_maxvar']], sigma=3)
#morphology.remove_small_objects(edges, min_size=np.pi*param.rbcR, in_place=True, connectivity=2)
#    
    
"""
Foreground segmentation
"""                   
#cent_2, label_mask_2 = segmentations.segment_fg_bg_onechannel_binary(im_corrected, mask_over, diag.measures['ch_maxvar'], vis_diag=True)   
hsv_resize, scale=imtools.imRescaleMaxDim(hsv_corrected,512,interpolation = 1)

cent_2, label_mask_2_resize = segmentations.segment_fg_bg_sv_kmeans4(hsv_resize, cent, vis_diag=vis_diag)   
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    label_mask_2 = img_as_ubyte(resize(label_mask_2_resize,( im.shape[0],im.shape[1]), order = 0))

# create foreground mask
mask_fg_sure=((label_mask_2==3)*255).astype('uint8')
imtools.maskOverlay(im,mask_fg_sure,0.5,vis_diag=vis_diag,fig='mask_fg_sure')

# remove holes from foreground mask

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mask_fg_sure_filled=img_as_ubyte(morphology.remove_small_holes(mask_fg_sure, min_size=param.rbcR*param.rbcR*np.pi, connectivity=4))

# opening
mask_fg_clear=255*morphology.binary_opening(mask_fg_sure_filled,morphology.disk(param.rbcR/4)).astype('uint8')
im_2=imtools.maskOverlay(im,mask_fg_clear,0.5,vis_diag=vis_diag,fig='mask_fg_sure_clear')

im_2=imtools.maskOverlay(im,mask_fg_clear,0.5,vis_diag=vis_diag)
imtools.maskOverlay(im_2,tmp,0.5,ch=0,vis_diag=vis_diag)

#TODO: loop over all connected components and investigate
#TODO: find median hue for RBC - check connected components for WBC

"""
Find cell markers - using dtf and local maximas
"""

# use dtf to find markers for watershed 
skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
# watershed seeds
local_maxi = feature.peak_local_max(dtf, indices=False, 
                                    threshold_abs=0.25*param.rbcR,
                                    footprint=np.ones((int(2*param.rbcR), int(2*param.rbcR))), 
                                    labels=mask_fg_clear.copy())
# Problem - similar maximas are left
# remove noisy maximas
markers = measure.label(local_maxi)
imtools.maskOverlay(im_2,255*morphology.binary_dilation((markers>0),morphology.disk(3)),0.5,ch=0,vis_diag=vis_diag,fig='markers') 
imtools.maskOverlay(im,255*morphology.binary_dilation((markers>0),morphology.disk(3)),0.5,ch=1,vis_diag=vis_diag,fig='markers') 
 
#TODO: count markers

# watershed on dtf
labels_ws = morphology.watershed(-dtf, markers, mask=mask_fg_clear)

# edge map for visualization
bounds=segmentation.find_boundaries(labels_ws,connectivity=3).astype('uint8')*255

im2=imtools.maskOverlay(im,bounds,0.5,ch=1,sbs=False,vis_diag=vis_diag,fig='contours')


"""
Save shapes
"""
#skimage.measure.regionprops

cnts = measure.find_contours(mask_fg_clear, 0.5)
regions = measure.regionprops(labels_ws)
#http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops

fc=plt.figure('contur image')
axc=fc.add_subplot(111)
axc.imshow(im)     

for n, contour in enumerate(cnts):
    axc.plot(contour[:,1], contour[:, 0], linewidth=2)
    axc.text(np.mean(contour[:,1]), np.mean(contour[:, 0]),str(n), bbox=dict(facecolor='white', alpha=0.5))

# TEST
l=imtools.normalize(label_mask_2,vis_diag=vis_diag,fig='kmeans_result')
rbcSize=param.rbcR*param.rbcR*np.pi
sat_maxclust=cent_2[:,0].max()
h_0=imtools.colorHist(diag.hsv,mask=mask_fg_clear,vis_diag=vis_diag,fig='all')
sat_cumh_0=np.add.accumulate(h_0[1]*1.25)


iBlob=70
ss=[]
fc=plt.figure('wbc image')
axc=fc.add_subplot(111)
axc.imshow(im)     

for iBlob, contour in enumerate(cnts):
    # TODO: check if not close to image border (mean_contour)
    # TODO: check if saturation histogram
        
    mask_blob=np.zeros((im.shape[0],im.shape[1]),'uint8')
    #contour=cnts[iBlob]
    rr,cc=skimage.draw.polygon(contour[:,0], contour[:, 1], shape=None)
    mask_blob[rr,cc]=255
    reg = measure.regionprops(mask_blob)
    r=reg[0]
    nEstimated=r.area/rbcSize
    print(r.convex_area/r.area)
    print(nEstimated)
    nMarkers=(local_maxi[mask_blob>0]>0).sum()
    print(nMarkers)
    h=imtools.colorHist(diag.hsv,mask=mask_blob,vis_diag=vis_diag,fig=str(iBlob))
    # check distribution for heavy tail
    sat_cumh=np.add.accumulate(h[1]*1.25)
    s=imtools.histogram_similarity(h_0[1],h[1])
    ss.append(s)
    if s<10:
        axc.plot(contour[:,1], contour[:, 0], linewidth=2)
    print(s)
    print(sat_cumh[sat_maxclust]) # wbc if this is small


     
         
imtools.maskOverlay(l,mask_blob,0.5,ch=1,sbs=False,vis_diag=True,fig='blob')

"""
Create SHAPES and store them
"""
# ToDo: merge groups
shapelist=[]
for c in cnts:
     one_shape=('RBC','general',c,'None','None')
     shapelist.append(one_shape)

head, tail=os.path.split(image_file)
xml_file=os.path.join(save_dir,tail.replace('.bmp',''))
tmp = annotations.AnnotationWriter(head,xml_file, (im.shape[0],im.shape[1]))
tmp.addShapes(shapelist)
tmp.save()

imtools.plotShapes(im,shapelist)
