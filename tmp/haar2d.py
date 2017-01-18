__author__ = 'SzMike'

import sys
import cv2
import pywt
import easygui
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import math
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def block_view(array, block= (3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (array.shape[0]/ block[0], array.shape[1]/ block[1])+ block
    strides= (block[0]* array.strides[0], block[1]* array.strides[1])+ array.strides
    return ast(array, shape= shape, strides= strides)

def bbImage(img,maxBBSize,nLevel,divisional):
    scale= maxBBSize/float(max(img.shape[0],img.shape[1]))   
    if scale<1:
        img_small=cv2.resize(img, (int(scale*img.shape[1]),int(scale*img.shape[0])), interpolation = cv2.INTER_AREA)
    else:
        img_small=img

    if img_small.ndim==2:
        img_small=img_small[:,:,np.newaxis]
        
    modSize=(img_small.shape[0]%(divisional*pow(2,nLevel)),img_small.shape[1]%(divisional*pow(2,nLevel)))
    img_bb=img_small[int(math.ceil(modSize[0]/2)):img_small.shape[0]-int(math.floor(modSize[0]/2)),int(math.ceil(modSize[1]/2)):img_small.shape[1]-int(math.floor(modSize[1]/2)),:]
   
    return img_bb
            

def haarImage(img,nLevel=3):
    imgHaar=np.empty(img.shape, dtype='float32')    
    coeffs = pywt.wavedec2(img, 'haar', level=nLevel)
    for i, coef in enumerate(coeffs): 
        if (i==0):
            imgHaar[0:coef.shape[0],0:coef.shape[1]]=coef/pow(2,3);
        else:
            imgHaar[coef[0].shape[0]:2*coef[0].shape[0],0:coef[0].shape[1]]=coef[0]/pow(2,(nLevel+1-i))
            imgHaar[0:coef[1].shape[0],coef[1].shape[1]:2*coef[1].shape[1]]=coef[1]/pow(2,(nLevel+1-i))
            imgHaar[coef[2].shape[0]:2*coef[2].shape[0],coef[2].shape[1]:2*coef[2].shape[1]]=coef[2]/pow(2,(nLevel+1-i))
    return imgHaar
    
def calcHaarEmap_oneLevel(haar1):
    # 3D, max among channels
    half_height = haar1.shape[0]/2;
    half_width  = haar1.shape[1]/2;

    imax=np.empty((half_height,half_width,3), dtype='float32') 

    imax[:,:,0]=abs(haar1[0:half_height,half_width:2*half_width,]).max(2) # horizontal
    imax[:,:,1]=abs(haar1[half_height:2*half_height,0:half_width,:]).max(2) # vertical
    imax[:,:,2]=abs(haar1[half_height:2*half_height,half_width:2*half_width,:]).max(2) # diagonal

    emap=abs(imax).max(2) # max over blocks
    return emap

def calcHaarEmap(haar,nLevel=3):
    (h, w, c) = haar.shape
    sh = h/(pow(2,nLevel))
    sw = w/(pow(2,nLevel))
    eMaps=np.empty((sh,sw,nLevel))    

    for iLevel in range(0,nLevel):
         sh = h/pow(2,iLevel)
         sw = w/pow(2,iLevel)
         tmp=calcHaarEmap_oneLevel(haar[0:sh,0:sw,:])
         eMaps[:,:,nLevel-iLevel-1]=block_view(tmp,
                                    block=(pow(2,(nLevel-iLevel-1)),pow(2,(nLevel-iLevel-1)))).max(2).max(2)
    
    return eMaps

def calcFocusMask(eMaps,edgeTsh):

    focusMask=np.empty(eMaps.shape, dtype='bool')    
    focusMask[:,:,0]=eMaps.min(2)>edgeTsh
    for iLevel in range(1,eMaps.shape[2]):
         focusMask[:,:,iLevel]= np.bitwise_and((eMaps[:,:,iLevel] >  eMaps[:,:,iLevel-1]),focusMask[:,:,iLevel-1]==1)
         
    return focusMask
    
def doMorphology(mask):
    r=int(max(mask.shape)/20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask
    



def main(photo_file):

    isGray=1   
    divisional=3.0
    nLevel=3
    residual=64.0
    edgeTsh=5
    maxBBSize=residual*pow(2,nLevel)*divisional;
    
    img_orig = cv2.imread(photo_file)

    if isGray:
        img_gray = cv2.cvtColor(img_orig.astype(dtype=np.uint8), cv2.COLOR_BGR2GRAY)
        img=np.expand_dims(img_gray, axis=2) 
        img[:,:,0]=img_gray
    else:
        img=img_orig
        
    #cv2.imshow('gray',image_gray) #.astype(dtype=np.uint8))
    #cv2.waitKey()

    img_bb=bbImage(img,maxBBSize,nLevel,divisional)   
    
    if isGray:
        img_bb3=np.repeat(img_bb, 3, axis=2)
    else:
        img_bb3=img_bb
        
    haar=np.empty(img_bb3.shape, dtype='float32')    

    for iCh in range(img_bb3.shape[2])  :  
        haar[:,:,iCh]=haarImage(img_bb3.astype(np.float32)[:,:,iCh],nLevel)
    
    eMaps=calcHaarEmap(haar,nLevel)
    
    focusMask=calcFocusMask(eMaps,edgeTsh).astype('uint8')
    fMask=doMorphology(focusMask[:,:,2])

    #cv2.imshow('fMask',255*focusMask[:,:,0]) #.astype(dtype=np.uint8))
    #cv2.waitKey()
    
    img_tmp=np.empty(focusMask.shape, dtype='uint8')   
    img_tmp.fill(0)
    img_tmp[:,:,0]=fMask
    cv2.normalize(img_tmp,img_tmp,255,0,cv2.NORM_MINMAX)
    
    img_foc = cv2.addWeighted(img_bb3,0.5,cv2.resize(img_tmp,(img_bb.shape[1],img_bb.shape[0])),0.5,0)    
        
    (h, w, c) = haar.shape
    sh = h/(pow(2,nLevel))
    sw = w/(pow(2,nLevel))
    img_small=haar[0:sh,0:sw,:]    

    cv2.imshow('fmap',cv2.resize(img_foc,(600,int((600/float(img_foc.shape[1])*img_foc.shape[0]))))) #.astype(dtype=np.uint8))
    cv2.waitKey()

if __name__ == '__main__':
    msg = 'Do you want to continue?'
    title = 'Please Confirm'
    photo_file=r'e:\Pictures\TestSets\Temp4\TUN_9789.jpg'
    #main(photo_file)
    while easygui.ccbox(msg, title):     # show a Continue/Cancel dialog
        photo_file = easygui.fileopenbox(default=photo_file)
        cv2.destroyAllWindows() # Ok, destroy the window
        main(photo_file)
    sys.exit(0)