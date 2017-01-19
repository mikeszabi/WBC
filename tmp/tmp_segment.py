# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:34:50 2017

@author: SzMike
"""
# from : http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker

import os
import numpy as np
import cv2
from skimage import segmentation
import matplotlib.pyplot as plt
%matplotlib qt5

image_dir=param.getTestImageDirs('Lymphocyte')
image_file=os.path.join(image_dir,'23.bmp')
im = cv2.imread(image_file,cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


img, regions = selective_search(
        im_orig, scale=0.2, sigma=0.8, min_size=10)

im_mask = segmentation.felzenszwalb(im_orig.astype('float64')/255, scale=30, sigma=0.8,min_size=300)

wsC = cv2.applyColorMap(segments.astype('uint8'), cv2.COLORMAP_PARULA)

cv2.imshow('alma',wsC)
cv2.waitKey()

mask=255*im_mask
mask=mask.astype('uint8')

tools.maskOverlay(im_orig,mask,0.5,1,1)

segments_slic = segmentation.slic(rgb.astype('float64')/255, n_segments=250, compactness=10, sigma=0.5)

cv2.imshow('alma',segmentation.mark_boundaries(im_orig.astype('float64')/255, segments_slic))
cv2.waitKey()

segments_quickshift = segmentation.quickshift(rgb.astype('float64')/255, kernel_size=3, max_dist=20, ratio=0.5)
tools.maskOverlay(im_orig,segments,0.5,1,1)

cv2.imshow('alma',segmentation.mark_boundaries(im_orig.astype('float64')/255, segments_quickshift))
cv2.waitKey()
