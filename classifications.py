# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:19:00 2017

@author: Szabolcs
"""

import os
import warnings

import numpy as np
from skimage import transform
from skimage import img_as_ubyte

from cntk import load_model

import cfg

def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)


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