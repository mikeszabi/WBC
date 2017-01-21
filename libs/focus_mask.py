# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:05:51 2017

@author: SzMike
"""
import pywt
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import math
import cv2


class focusMask:
    def __init__(self):
        self.isGray=1    # in microns
        self.divisional=3.0
        self.nLevel=3
        self.residual=64.0
        self.edgeTsh=5
        self.maxBBSize=self.residual*pow(2,self.nLevel)*self.divisional;

    def bbImage(self,img):
        self.scale= self.maxBBSize/float(max(img.shape[0],img.shape[1]))   
        if self.scale<1:
            img_small=cv2.resize(img, (int(self.scale*img.shape[1]),int(self.scale*img.shape[0])), interpolation = cv2.INTER_AREA)
        else:
            img_small=img
    
        if img_small.ndim==2:
            img_small=img_small[:,:,np.newaxis]
            
        modSize=(img_small.shape[0]%(self.divisional*pow(2,self.nLevel)),img_small.shape[1]%(self.divisional*pow(2,self.nLevel)))
        img_bb=img_small[int(math.ceil(modSize[0]/2)):img_small.shape[0]-int(math.floor(modSize[0]/2)),int(math.ceil(modSize[1]/2)):img_small.shape[1]-int(math.floor(modSize[1]/2)),:]
       
        if (self.isGray) and (img_bb.shape[2]==3):
            img_gray = cv2.cvtColor(img_bb.astype(dtype=np.uint8), cv2.COLOR_BGR2GRAY)
            img=np.expand_dims(img_gray, axis=2) 
            img[:,:,0]=img_gray
        else:
            img=img_bb
        
        return img
    
    def block_view(self,array, block= (3, 3)):
        """Provide a 2D block view to 2D array. No error checking made.
        Therefore meaningful (as implemented) only for blocks strictly
        compatible with the shape of A."""
        # simple shape and strides computations may seem at first strange
        # unless one is able to recognize the 'tuple additions' involved ;-)
        shape= (array.shape[0]/ block[0], array.shape[1]/ block[1])+ block
        strides= (block[0]* array.strides[0], block[1]* array.strides[1])+ array.strides
        return ast(array, shape= shape, strides= strides)
    
    def haarImage(self,img):
        imgHaar=np.empty(img.shape, dtype='float32')    
        coeffs = pywt.wavedec2(img, 'haar', level=self.nLevel)
        for i, coef in enumerate(coeffs): 
            if (i==0):
                imgHaar[0:coef.shape[0],0:coef.shape[1]]=coef/pow(2,3);
            else:
                imgHaar[coef[0].shape[0]:2*coef[0].shape[0],0:coef[0].shape[1]]=coef[0]/pow(2,(self.nLevel+1-i))
                imgHaar[0:coef[1].shape[0],coef[1].shape[1]:2*coef[1].shape[1]]=coef[1]/pow(2,(self.nLevel+1-i))
                imgHaar[coef[2].shape[0]:2*coef[2].shape[0],coef[2].shape[1]:2*coef[2].shape[1]]=coef[2]/pow(2,(self.nLevel+1-i))
        return imgHaar
        
    def calcHaarEmap_oneLevel(self,haar1):
        # 3D, max among channels
        half_height = int(haar1.shape[0]/2);
        half_width  = int(haar1.shape[1]/2);
    
        imax=np.empty((half_height,half_width,3), dtype='float') 
    
        imax[:,:,0]=abs(haar1[0:half_height,half_width:2*half_width,]).max(2) # horizontal
        imax[:,:,1]=abs(haar1[half_height:2*half_height,0:half_width,:]).max(2) # vertical
        imax[:,:,2]=abs(haar1[half_height:2*half_height,half_width:2*half_width,:]).max(2) # diagonal
    
        emap=abs(imax).max(2) # max over blocks
        return emap
    
    def calcHaarEmap(self,haar):
        # nLevel has to be 3
        (h, w, c) = haar.shape
        sh = int(h/(pow(2,self.nLevel)))
        sw = int(w/(pow(2,self.nLevel)))
        eMaps=np.empty((sh,sw,self.nLevel),dtype='float')  
    
        for iLevel in range(0,self.nLevel):
             sh = h/pow(2,iLevel)
             sw = w/pow(2,iLevel)
             tmp=self.calcHaarEmap_oneLevel(haar[0:int(sh),0:int(sw),:])
             eMaps[:,:,self.nLevel-iLevel-1]=self.block_view(tmp,
                                        block=(pow(2,(self.nLevel-iLevel-1)),pow(2,(self.nLevel-iLevel-1)))).max(2).max(2)
        
        return eMaps
    
    def calcFocusMask(self,eMaps):
    
        focusMask=np.empty(eMaps.shape, dtype='bool')    
        focusMask[:,:,0]=eMaps.min(2)>self.edgeTsh
        for iLevel in range(1,eMaps.shape[2]):
             focusMask[:,:,iLevel]= np.bitwise_and((eMaps[:,:,iLevel] >  eMaps[:,:,iLevel-1]),focusMask[:,:,iLevel-1]==1)
             
        return focusMask