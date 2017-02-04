# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:33:32 2017

@author: SzMike
"""

import cv2
import numpy as np;
import tools
from skimage.color import im_s2hsv, im_s2gray
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import random


def overMask(im):
    if len(im.shape)==3:
        # im_s image
        gray=np.floor(rgb2gray(im)*256)
        gray=gray.astype('uint')
    else:
        gray=im
    overexpo_mask=np.empty(gray.shape, dtype='bool') 
    overexpo_mask=gray==255
    overexpo_mask=255*overexpo_mask.astype(dtype=np.uint8) 
    return overexpo_mask


def illumination_inhomogenity(im, bg_mask, vis_diag):
    if len(im.shape)==3:
        # im_s image
        gray=np.floor(rgb2gray(im)*256)
        gray=gray.astype('uint')
    else:
        gray=im_s2gray
    
    
    gray[bg_mask==0]=0
    gray_s, scale=tools.imresizeMaxDim(gray, 64, interpolation = cv2.INTER_NEAREST)
    mask=gray_s==0
    mask=255*(mask.astype('uint8'))
    inpainted = cv2.inpaint(gray_s, mask, 11, cv2.INPAINT_NS)
    inpainted=inpainted[3:-3,3:-3]
    inpainted = cv2.medianBlur(inpainted,9)
    tools.normalize(inpainted,vis_diag,fig='expositionmap')
    return cv2.resize(inpainted, (gray.shape), interpolation = cv2.INTER_LINEAR)

    
def segment(im, vis_diag=False):
    
    
    # segmentation on hsv image
    
    #param=cfg.param()
    # TODO: use parameters from cfg
    
    # create small image
    im_s, scale = tools.imresizeMaxDim(im, 256)
    
    hsv = np.floor(rgb2hsv(im_s)*256)
    hsv = hsv.astype('uint8')            
    # TODO: 256 as parameter

    # overexpo mask
    overexpo_mask=overMask(hsv[:,:,2])
    
    #hist = tools.colorHist(hsv[:,:,2],vis_diag)
    #tools.maskOverlay(hsv,overexpo_mask,0.5,1,sbs=False,vis_diag=vis_diag)

    # KMEANS on saturation and intensity
    Z = hsv.reshape((-1,3))
    Z = np.float32(Z)/256
                  
    # mask with overexpo
    Z_mask=overexpo_mask.reshape((-1,1))==0
    Z_mask=Z_mask.flatten()

    # select saturation and value channels
    Z_1=Z[Z_mask,1:3]
    Z=Z[:,1:3]

    kmeans = KMeans(n_clusters=3, random_state=0).fit(Z_1)
    
    # TODO: initialize centers from histogram peaks
    center = np.uint8(kmeans.cluster_centers_*256)
    label = kmeans.labels_
    #print(center)

    if vis_diag:
        rs=random.sample(range(0, Z_1.shape[0]-1), 1000)
        Z_rs=Z_1[rs,:]

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111)
        plt.cla()
        label_rs = label[rs]
        ax.scatter(Z_rs[:, 0], Z_rs[:, 1], c=label_rs.astype(np.float))


        ax.set_xlabel('Saturation')
        ax.set_ylabel('Value')
        for c in center:
            ax.plot(c[0]/255, c[1]/255, 'o',markeredgecolor='k', markersize=15)

        plt.show()


    lab_all=np.zeros(Z.shape[0])
    lab_all.flat[Z_mask==False]=-1
    lab_all.flat[Z_mask==True]=label
    tools.normalize(lab_all.reshape((hsv.shape[0:2])),vis_diag=vis_diag,fig='fg_bg_labels')
            
       
    # not overexposed mask
    maxi=np.argmax(center[:,1])
    sure_bg_mask = lab_all.reshape((hsv.shape[0:2]))==maxi
    sure_bg_mask = tools.normalize(sure_bg_mask.astype('uint8'),vis_diag=vis_diag,fig='sure_bg')

    maxi=np.argmax(center[:,0])
    sure_fg_mask = lab_all.reshape((hsv.shape[0:2]))==maxi
    sure_fg_mask = tools.normalize(sure_fg_mask.astype('uint8'),vis_diag=vis_diag,fig='sure_fg')

    unsure_mask = np.ones(sure_fg_mask.shape)
    unsure_mask[np.logical_or(np.logical_or(sure_fg_mask,sure_bg_mask),overexpo_mask)]=0
    unsure_mask = tools.normalize(unsure_mask.astype('uint8'),vis_diag=vis_diag,fig='unsure')

    masks=np.zeros((hsv.shape[0],hsv.shape[1],4)).astype('uint8')
    masks[:,:,0]=overexpo_mask
    masks[:,:,1]=sure_fg_mask
    masks[:,:,2]=sure_bg_mask
    masks[:,:,3]=unsure_mask
         
    masks = cv2.resize(masks, (hsv_orig.shape[1],hsv_orig.shape[0]), interpolation = cv2.INTER_NEAREST)

    
    return center, masks


