# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""

import os
import glob
import warnings

import cv2
import skimage.io as io
import numpy as np;
from skimage.transform import resize
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
from skimage.color import rgb2hsv, hsv2rgb
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_dilation, disk, \
                                medial_axis, \
                                remove_small_objects, remove_small_holes

import matplotlib.pyplot as plt
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
vis_diag=True

imDirs=os.listdir(param.getImageDirs(''))
print(imDirs)
image_dir=param.getImageDirs(imDirs[1])
print(glob.glob(os.path.join(image_dir,'*.bmp')))
image_file=os.path.join(image_dir,'420.bmp')

# reading image
im = io.imread(image_file) # read uint8 image

if vis_diag:
    fo=plt.figure('original image')
    axo=fo.add_subplot(111)
    axo.imshow(im)              
              
# diagnose image
diag=diagnostics.diagnostics(im,image_file,vis_diag=True)

"""
inhomogen illumination correction
"""

# create sure background with n=4 kmeans, on sv channel
cent, label_mask_small = segmentations.segment_fg_bg_sv_kmeans4(diag.hsv_small, 'k-means++', vis_diag=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    label_mask = img_as_ubyte(resize(label_mask_small,( im.shape[0],im.shape[1]), order = 0))

mask_bg_sure= np.logical_or(label_mask == 0, label_mask == 1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hsv_corrected= diagnostics.illumination_inhomogenity(diag.hsv, mask_bg_sure, vis_diag=True)
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
if vis_diag:
    tmp=imtools.normalize(img_as_ubyte(edge_mag),vis_diag=True)
##    
##threshold_global_otsu = threshold_otsu(edge_mag)
##edges = edge_mag >= threshold_global_otsu
#edges = feature.canny(hsv_corrected[:,:,diag.measures['ch_maxvar']], sigma=3)
#remove_small_objects(edges, min_size=np.pi*param.rbcR, in_place=True, connectivity=2)
#    
    
"""
background-foreground segmentation
"""                   
#cent_2, label_mask_2 = segmentations.segment_fg_bg_onechannel_binary(im_corrected, mask_over, diag.measures['ch_maxvar'], vis_diag=True)   
hsv_resize, scale=imtools.imRescaleMaxDim(hsv_corrected,512,interpolation = 1)

cent_2, label_mask_2_resize = segmentations.segment_fg_bg_sv_kmeans4(hsv_resize, cent, vis_diag=True)   
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    label_mask_2 = img_as_ubyte(resize(label_mask_2_resize,( im.shape[0],im.shape[1]), order = 0))

# create foreground mask
sure_fg_mask=((label_mask_2==3)*255).astype('uint8')
imtools.maskOverlay(im,sure_fg_mask,0.5,vis_diag=True,fig='sure_fg_mask')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rbc_mask=img_as_ubyte(remove_small_holes(sure_fg_mask, min_size=param.rbcR*param.rbcR*np.pi, connectivity=4))

# opening
fg_mask_open=255*binary_opening(rbc_mask,disk(param.rbcR/4)).astype('uint8')
im_2=imtools.maskOverlay(im,fg_mask_open,0.5,vis_diag=True,fig='sure_fg_mask_filled')

im_2=imtools.maskOverlay(im,fg_mask_open,0.5,vis_diag=True)
imtools.maskOverlay(im_2,tmp,0.5,ch=0,vis_diag=True)

#TODO: loop over all connected components and investigate
#TODO: find median hue for RBC - check connected components for WBC

 # use dtf to find markers for watershed
#fg_mask_open_with_edges=255*(np.logical_and(fg_mask_open,np.logical_not(edges))).astype('uint8')
 
skel, dtf = medial_axis(fg_mask_open, return_distance=True)
dtf.flat[(fg_mask_open>0).flatten()]+=np.random.random(((fg_mask_open>0).sum()))
# watershed seeds
local_maxi = feature.peak_local_max(dtf, indices=False, 
                                    threshold_abs=0.25*param.rbcR,
                                    footprint=np.ones((int(2*param.rbcR), int(2*param.rbcR))), 
                                    labels=fg_mask_open.copy())
# Problem - similar maximas are left
# remove noisy maximas
markers = measure.label(local_maxi)
imtools.maskOverlay(im_2,255*binary_dilation((markers>0),disk(3)),0.5,ch=0,vis_diag=True,fig='markers') 
 
#TODO: count markers

# watershed on dtf
labels_ws = morphology.watershed(-dtf, markers, mask=fg_mask_open)

# edge map for visualization
mag=segmentation.find_boundaries(labels_ws).astype('uint8')*255

im2=imtools.maskOverlay(im,mag,0.5,ch=1,sbs=False,vis_diag=True,fig='contours')
# counting



"""
Save shapes
"""


# ToDo: merge groups
shapelist=[]
for label in np.unique(labels_ws):
    	# if the label is zero, we are examining the 'background'
    	# so simply ignore it
     if label == 0:
         continue
  
     mask = np.zeros(labels_ws.shape, dtype="uint8")
     mask[labels_ws == label] = 255
     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
     c = max(cnts, key=cv2.contourArea)
     one_shape=('RBC','general',c.reshape(c.shape[0],c.shape[2]),'None','None')
     shapelist.append(one_shape)
#     x,y,w,h = cv2.boundingRect(c)
     #TODO: túl nagy pacnik további vizsgálata
#     if ((x>param.rbcR) & (x+w<im.shape[1]-param.rbcR) & 
#         (y>param.rbcR) & (y+h<im.shape[0]-param.rbcR)):
#        cv2.rectangle(im2,(x,y),(x+w,y+h),(255,255,255),2)
#        cv2.putText(im2, "#{}".format(label), (x - 10, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
#        if cv2.contourArea(c)>2*int(math.pi*math.pow(param.wbcRatio*param.rbcR,2)):
#            cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),3)
#fh=plt.figure('detected')
#ax=fh.add_subplot(111)
#ax.imshow(im2)

head, tail=os.path.split(image_file)
tmp = annotations.AnnotationWriter(head,tail.replace('.bmp',''), (im.shape[0],im.shape[1]))
tmp.addShapes(shapelist)
tmp.save()