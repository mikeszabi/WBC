# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:33:32 2017

@author: SzMike
"""

import warnings
import cv2
import numpy as np;
import tools
from skimage import img_as_ubyte
from skimage.color import rgb2hsv, rgb2gray
from skimage.restoration import inpaint
from skimage.morphology import disk
from skimage.filters.rank import median
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import random


def overMask(im):
    if len(im.shape)==3:
        # im_s image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gray=img_as_ubyte(rgb2gray(im))
    else:
        gray=im
    overexpo_mask=np.empty(gray.shape, dtype='bool') 
    overexpo_mask=gray==255
    overexpo_mask=255*overexpo_mask.astype(dtype=np.uint8) 
    return overexpo_mask


def illumination_inhomogenity(im, bg_mask, vis_diag):
    # using inpainting techniques
    assert im.dtype=='uint8', 'Not uint8 type'
    
    if len(im.shape)==3:
        # im_s image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gray=rgb2gray(im)
    else:
        gray=im
    
    
    gray[bg_mask==0]=0
    gray_s, scale=tools.imresizeMaxDim(gray, 64, interpolation = 0)
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask=img_as_ubyte(gray_s==0)
    inpainted =  inpaint.inpaint_biharmonic(gray_s, mask, multichannel=False)
    inpainted = median(inpainted, disk(15))
    tools.normalize(inpainted,vis_diag,fig='expositionmap')
# TODO: show the unnormalized inhomogenity on full 0-255 range    
    return cv2.resize(inpainted, (gray.shape), interpolation = 1)

    
def segment(im, vis_diag=False):
    
    
    # segmentation on hsv image
    
    #param=cfg.param()
    # TODO: use parameters from cfg
    
    # create small image
    im_s, scale = tools.imresizeMaxDim(im, 256, interpolation=1)
    
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hsv = img_as_ubyte(rgb2hsv(im_s))

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
    lab_all=lab_all.reshape((hsv.shape[0:2]))
    tools.normalize(lab_all,vis_diag=vis_diag,fig='fg_bg_labels')
          
    
    # adding meaningful labels
    lab=np.zeros(lab_all.shape).astype('uint8')
    
    # sure background mask - smallest saturation
    mins=np.argmin(center[:,0]) # smallest saturation
    lab[lab_all==mins]=1
                 
    # sure foreground mask -largest saturation
    maxs=np.argmax(center[:,0]) # largest saturation
    lab[lab_all==maxs]=3
   
    uns=0
    for i in range(3):
       if i not in (mins,maxs):
           uns=i
           
    # unsure region
    lab[lab_all==uns]=2   
    lab = cv2.resize(lab, (im.shape[1],im.shape[0]), interpolation = 0)

    # lab == 0 : overexposed
    # lab == 1 : sure bckg
    # lab ==2 : unsure
    # lab == 3 : sure foreground

    return center[[mins,uns,maxs],:], lab


