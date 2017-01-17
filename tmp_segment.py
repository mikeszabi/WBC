# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:34:50 2017

@author: SzMike
"""

import os
import numpy as np
from defPaths import *
from selectivesearch import selective_search
import cv2
import skimage.segmentation
import tools


image_file=os.path.join(image_dir,'60.bmp')
im_orig = cv2.imread(image_file,cv2.IMREAD_COLOR)


img, regions = selective_search(
        im_orig, scale=0.2, sigma=0.8, min_size=10)

im_mask = skimage.segmentation.felzenszwalb(im_orig.astype('float64'), scale=3, sigma=0.8,min_size=10)

mask=255*im_mask
mask=mask.astype('uint8')

tools.maskOverlay(im_orig,mask,0.5,1,1)
