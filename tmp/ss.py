# -*- coding: utf-8 -*-g
"""
Created on Thu Jan 26 14:52:10 2017

@author: SzMike
"""

import _init_path
_init_path.add_path('tmp')
import cv2
from cntk_helpers import *

from selectivesearch import selective_search

import matplotlib.pyplot as plt
%matplotlib qt5


def imresize(img, scale, interpolation = cv2.INTER_LINEAR):
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)


def imresizeMaxDim(img, maxDim, boUpscale = False, interpolation = cv2.INTER_LINEAR):
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1  or boUpscale:
        img = imresize(img, scale, interpolation)
    else:
        scale = 1.0
    return img, scale

def filterRois(rects, maxWidth, maxHeight, roi_minNrPixels, roi_maxNrPixels,
               roi_minDim, roi_maxDim, roi_maxAspectRatio):
    filteredRects = []
    filteredRectsSet = set()
    for rect in rects:
        if tuple(rect) in filteredRectsSet: # excluding rectangles with same co-ordinates
            continue

        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y
        assert(w>=0 and h>=0)

        # apply filters
        if h == 0 or w == 0 or \
           x2 > maxWidth or y2 > maxHeight or \
           w < roi_minDim or h < roi_minDim or \
           w > roi_maxDim or h > roi_maxDim or \
           w * h < roi_minNrPixels or w * h > roi_maxNrPixels or \
           w / h > roi_maxAspectRatio or h / w > roi_maxAspectRatio:
               continue
        filteredRects.append(rect)
        filteredRectsSet.add(tuple(rect))

    # could combine rectangles using non-maxima surpression or with similar co-ordinates
    # groupedRectangles, weights = cv2.groupRectangles(np.asanyarray(rectsInput, np.float).tolist(), 1, 0.3)
    # groupedRectangles = nms_python(np.asarray(rectsInput, np.float), 0.5)
    assert(len(filteredRects) > 0)
    return filteredRects

filename=r'd:\Projects\data\GROCERY\positive\WIN_20160803_12_38_42_Pro.jpg'
imgOrig = cv2.imread(filename,cv2.IMREAD_COLOR)

img, scale = imresizeMaxDim(imgOrig, 200, boUpscale=True, interpolation = cv2.INTER_AREA)
_, ssRois = selective_search(img, scale=100, sigma=1.2, min_size=20)
rects=[]
for ssRoi in ssRois:
    x, y, w, h = ssRoi['rect']
    rects.append([x,y,x+w,y+h])

rois = filterRois(rects, img.shape[0], img.shape[1], 20, 100000, 5, 200, 10)

img_rect=img.copy()
for r in rois:
    cv2.rectangle(img_rect,(r[0],r[1]),(r[2],r[3]),(255,255,255),1)

plt.imshow(img_rect)
