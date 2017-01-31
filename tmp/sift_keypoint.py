# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:00:18 2017

@author: SzMike
"""

import cv2

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(im_onech,None)

kp_f=[]
for i, k in enumerate(kp):
    if k.size>20 and k.size<40:
        kp_f.append(k)

im2=im.copy()
im2=cv2.drawKeypoints(im,kp_f,im2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(im2)
