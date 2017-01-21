# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:05:01 2017

@author: SzMike
"""

import os
import _init_path

import focus_mask
from focus_mask import focusMask
from params import param
import numpy as np

param=param()
focus_mask=focusMask()

def doMorphology(mask):
    r=int(max(mask.shape)/50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask    

imDirs=os.listdir(param.getTestImageDirs(''))
print(imDirs)
image_dir=param.getTestImageDirs(imDirs[4])
image_file=os.path.join(image_dir,'34.bmp')
im = cv2.imread(image_file,cv2.IMREAD_COLOR)
im_onech = im[:,:,1];
    
fMask=focusMask()

img_bb=fMask.bbImage(im_onech)   


if fMask.isGray:
    img_bb3=np.repeat(img_bb, 3, axis=2)
else:
    img_bb3=img_bb
    
haar=np.empty(img_bb3.shape, dtype='float')    

for iCh in range(img_bb3.shape[2])  :  
    haar[:,:,iCh]=fMask.haarImage(img_bb3.astype(np.float32)[:,:,iCh])

eMaps=fMask.calcHaarEmap(haar)

mask=fMask.calcFocusMask(eMaps).astype('uint8')

f2Mask=doMorphology(mask[:,:,0])

img_tmp=np.empty(mask.shape, dtype='uint8')   
img_tmp.fill(0)
img_tmp[:,:,0]=f2Mask
cv2.normalize(img_tmp,img_tmp,255,0,cv2.NORM_MINMAX)
img_foc = cv2.addWeighted(img_bb3,0.3,cv2.resize(img_tmp,(img_bb.shape[1],img_bb.shape[0])),0.7,0)    
        
cv2.imshow('focus',img_foc)
cv2.waitKey()