# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:58:52 2017

@author: SzMike
"""

# Standard imports
import cv2
import numpy as np;
from defPaths import *
import haar2d
 
image_file=os.path.join(image_dir,'36.bmp')

# Read imageű
im = cv2.imread(image_file) #, cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
params.minArea = 16
params.maxArea = 100000000
params.filterByArea = True
params.thresholdStep=5
detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

haar2d.main(image_file)