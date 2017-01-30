# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:34:50 2017

@author: SzMike
"""
# from : http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker

import os
import _init_path
import numpy as np
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
import math

import cv2

import matplotlib.pyplot as plt
%matplotlib qt5

from params import param
import tools

param=param()
    
imDirs=os.listdir(param.getImageDirs(''))
print(imDirs)
image_dir=param.getImageDirs(imDirs[0])
image_file=os.path.join(image_dir,'4.bmp')
im = cv2.imread(image_file,cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

scale= 200/float(max(rgb.shape[0],rgb.shape[1]))   
rgb_small=cv2.resize(rgb, (int(scale*rgb.shape[1]),int(scale*rgb.shape[0])), interpolation = cv2.INTER_AREA)


im_mask = segmentation.felzenszwalb(rgb_small.astype('float64')/255, scale=30, sigma=1.2, min_size=20)
tools.maskOverlay(rgb_small,im_mask.astype('uint8'),0.5,1,1)

wsC = cv2.applyColorMap(im_mask.astype('uint8'), cv2.COLORMAP_PARULA)

cv2.imshow('alma',wsC)
cv2.waitKey()

mask=255*im_mask
mask=mask.astype('uint8')

tools.maskOverlay(rgb_small,mask,0.5,1,1)

segments_slic = segmentation.slic(rgb.astype('float64')/255, n_segments=250, compactness=10, sigma=0.5)

cv2.imshow('alma',segmentation.mark_boundaries(im_orig.astype('float64')/255, segments_slic))
cv2.waitKey()

segments_quickshift = segmentation.quickshift(rgb.astype('float64')/255, kernel_size=3, max_dist=20, ratio=0.5)
tools.maskOverlay(im_orig,segments,0.5,1,1)

cv2.imshow('alma',segmentation.mark_boundaries(im_orig.astype('float64')/255, segments_quickshift))
cv2.waitKey()
