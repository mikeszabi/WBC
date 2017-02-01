# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:33:32 2017

@author: SzMike
"""

import cv2
import numpy as np;
import cfg
import tools


def overMask(intensity_image):
    overexpo_mask=np.empty(intensity_image.shape, dtype='bool') 
    overexpo_mask=intensity_image==255
    overexpo_mask=255*overexpo_mask.astype(dtype=np.uint8) 
    return overexpo_mask


def inhomogen(intensity_image, sure_bg_mask, mean_background_intensity, plotFlag):
    illumination_inhomogenity=intensity_image.astype('float')-mean_background_intensity.astype('float')
    illumination_inhomogenity.flat[sure_bg_mask.flatten==False]=0
    tools.normalize(illumination_inhomogenity,plotFlag)
    return illumination_inhomogenity
    
def segment(hsv_orig, plotFlag=False):
    # segmentation on hsv image
    
    #param=cfg.param()
    # TODO: use parameters from cfg
    
    # create small image
    hsv, scale = tools.imresizeMaxDim(hsv_orig, 256)
    # TODO: 256 as parameter

    # KMEANS on saturation and value
    Z = hsv.reshape((-1,3))

    # overexpo mask
    overexpo_mask=overMask(hsv[:,:,2])
    
    #hist = tools.colorHist(hsv,plotFlag,mask=255-overexpo_mask)
    #tools.maskOverlay(rgb,overexpo_mask,0.5,1,sbs=False,plotFlag=plotFlag)

    # KMEANS on saturation and intensity
    Z = hsv.reshape((-1,3))
    Z = np.float32(Z)/256
                  
    # mask with overexpo
    Z_mask=overexpo_mask.reshape((-1,1))==0
    Z_mask=Z_mask.flatten()

    # select saturation and value channels
    Z_1=Z[Z_mask,1:3]
    Z=Z[:,1:3]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    # TODO: why 3 ?
    ret,label,center=cv2.kmeans(Z_1,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    # TODO: initialize centers from histogram peaks
    center = np.uint8(center*256)
    #print(center)

    lab_all=np.zeros(Z.shape[0])
    lab_all.flat[Z_mask==False]=-1
    lab_all.flat[Z_mask==True]=label
    tools.normalize(lab_all.reshape((hsv.shape[0:2])),plotFlag=plotFlag)
            
       
    # not overexposed mask
    maxi=np.argmax(center[:,1])
    sure_bg_mask = lab_all.reshape((hsv.shape[0:2]))==maxi
    sure_bg_mask = tools.normalize(sure_bg_mask.astype('uint8'),plotFlag=plotFlag)

    maxi=np.argmax(center[:,0])
    sure_fg_mask = lab_all.reshape((hsv.shape[0:2]))==maxi
    sure_fg_mask = tools.normalize(sure_fg_mask.astype('uint8'),1)

    unsure_mask = np.ones(sure_fg_mask.shape)
    unsure_mask[np.logical_or(np.logical_or(sure_fg_mask,sure_bg_mask),overexpo_mask)]=0
    unsure_mask = tools.normalize(unsure_mask.astype('uint8'),plotFlag=plotFlag)

    masks=np.zeros((hsv.shape[0],hsv.shape[1],4)).astype('uint8')
    masks[:,:,0]=overexpo_mask
    masks[:,:,1]=sure_fg_mask
    masks[:,:,2]=sure_bg_mask
    masks[:,:,3]=unsure_mask
         
    masks = cv2.resize(masks, (hsv_orig.shape[1],hsv_orig.shape[0]), interpolation = cv2.INTER_NEAREST)

    
    return center, masks


