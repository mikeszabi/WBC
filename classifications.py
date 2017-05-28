# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:19:00 2017

@author: Szabolcs
"""

import os
import warnings

import cv2
import numpy as np
from skimage import exposure
from skimage import transform
from skimage import img_as_ubyte

from cntk import load_model

import cfg

def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def crop_shape(im,mask,one_shape,rgb_norm,med_rgb,scale=1,adjust=True):
    # one_shape is a detected shape on im - it's elemnets are: (cell_type,polygon_type,pts,'None','None')
    mins=(np.min(one_shape[2],axis=0)*scale).astype('int32')
    maxs=(np.max(one_shape[2],axis=0)*scale).astype('int32')
    o=(mins+maxs)/2
    r=(maxs-mins)/2
        
    if min(mins)>=0 and maxs[1]<im.shape[0] and maxs[0]<im.shape[1]:
        # loop over angles
        im_rotated=rotate(im,0,center=(o[0],o[1]))
        im_cropped=im_rotated[max(mins[1],0):min(maxs[1],im.shape[0]-1),\
                            max(mins[0],0):min(maxs[0],im.shape[1]-1)]
        mask_cropped=mask[max(mins[1],0):min(maxs[1],im.shape[0]-1),\
                            max(mins[0],0):min(maxs[0],im.shape[1]-1)]
    
        if adjust:
            # Normalization
            # local to cropped
#            pixs=im_cropped[mask_cropped>0,]
#            nuc_med_rgb=np.median(pixs,axis=0)
            # global to image
#            gamma=np.zeros(3)
#            gain=np.zeros(3)
#            for ch in range(3):
#                gamma[ch]=np.log(255-rgb_norm[ch])/np.log(255-med_rgb[ch])
#                gain[ch]=rgb_norm[ch]/np.power(med_rgb[ch],gamma[ch])
# gamma and gain ONLY FOR BLUE CHANNEL
            gamma=np.log(255-rgb_norm[2])/np.log(255-med_rgb[2])
            gain=min(255/im_cropped.max(),rgb_norm[2]/np.power(med_rgb[2],gamma))
            im_cropped=exposure.adjust_gamma(im_cropped,gamma=np.mean(gamma),gain=np.mean(gain))
   
    else:
        im_cropped=None
        mask_cropped=None
        
    return im_cropped, mask_cropped, o, r

class cnn_classification:
    def __init__(self):
        # model specific parameters
        self.param=cfg.param()
        self.img_size=32
        self.img_mean=128
        self.model_name='cnn_model.dnn'
        model_file=os.path.join(self.param.model_dir,self.model_name)
        print('...loading classification model')
        self.pred=load_model(model_file)
    
    def classify(self, im_cropped):
        # data--im_cropped
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = img_as_ubyte(transform.resize(im_cropped, (self.img_size,self.img_size), order=1))
        #data = 255*transform.resize(im_cropped, (self.img_size,self.img_size), order=1)
        rgb_image=data.astype('float32')
        rgb_image  -= self.img_mean
        bgr_image = rgb_image[..., [2, 1, 0]]
        pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

       
        result  = np.round(np.squeeze(self.pred.eval({self.pred.arguments[0]:[pic]}))*100)
        maxi=np.argmax(result)
        predicted_label=keysWithValue(self.param.wbc_basic_types,str(maxi))
        
        return predicted_label, result[maxi]