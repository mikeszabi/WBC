# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:13:50 2017

@author: SzMike
"""

import warnings
import csv
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.transform import resize
from skimage.restoration import inpaint
from skimage.filters import gaussian
from skimage import color
import skimage.io as io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imtools
import cfg


class diagnostics:
    def __init__(self,im,image_file,vis_diag=False,write=False):
        param=cfg.param()
        self.image_file=image_file
        assert len(im.shape)==3, 'Not 3 dimensional data'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.csp = img_as_ubyte(color.rgb2hsv(im))
            l_dim=2 # luminosity dimension
            self.csp_small, scale = imtools.imRescaleMaxDim(self.csp, param.small_size, interpolation=0)
            self.intensity_im=self.csp[:,:,l_dim]
            self.mask_over=self.overMask(self.intensity_im)
            self.mask_over_small, scale=imtools.imRescaleMaxDim(self.mask_over, param.small_size, interpolation=0)
        if vis_diag:
            imtools.maskOverlay(im,self.mask_over,0.5,vis_diag=True,fig='overexposition mask')
        
        self.measures={}
        self.imhists=imtools.colorHist(im,mask=255-self.mask_over,vis_diag=False,fig='rgb')
        self.csphists=imtools.colorHist(self.csp,mask=255-self.mask_over,vis_diag=False,fig='csp')
        
        self.cumh_rgb, siqr_rgb = self.semi_IQR(self.imhists) # Semi-Interquartile Range
        self.cumh_csp, siqr_csp = self.semi_IQR(self.csphists) # Semi-Interquartile Range

        minI=np.argwhere(self.cumh_csp[l_dim]>0.05)[0,0]
        maxI=np.argwhere(self.cumh_csp[l_dim]>0.95)[0,0]
                                                                          
        self.measures['siqr_rgb']=siqr_rgb
        self.measures['siqr_csp']=siqr_csp
        self.measures['ch_maxvar']=np.argmax(self.measures['siqr_rgb']) # channel with maximal variability
        self.measures['maxI']=maxI.astype('float64')
        self.measures['minI']=minI.astype('float64')
        self.measures['contrast']=(self.measures['maxI']-self.measures['minI'])/(self.measures['maxI']+self.measures['minI'])
        self.measures['overexpo_pct']=(self.mask_over>0).sum()/self.mask_over.size
        self.measures['global_entropy']=np.NaN # global homogenity
        self.measures['global_var']=np.NaN # global variance
        self.measures['saturation_pcts']=np.NaN # Todo: csp
        
        self.error_list=[]
        self.checks()

    def checks(self):
        self.error_list=[]
        if np.logical_not(self.measures['ch_maxvar']==1):
            self.error_list.append('ch_maxvar')
        if self.measures['contrast']<0.25:
            self.error_list.append('contrast')
        if self.measures['overexpo_pct']>0.1:
            self.error_list.append('overexpo_pct')
        print('Error list:')
        for errors in self.error_list:
            print(errors+' :'+str(self.measures[errors]))

    def semi_IQR(self,hists):
        siqr=[]
        cumh=[]
        for ch, h in enumerate(hists):
           h_norm=h/h.sum()
           cumh.append(np.add.accumulate(h_norm))
           Q1=np.argwhere(cumh[ch]>0.25)[0,0]
           Q3=np.argwhere(cumh[ch]>0.75)[0,0]
           siqr.append((Q3-Q1)/2)
        return cumh, siqr

    def overMask(self,intensity_im):
        assert intensity_im.dtype=='uint8', 'Not uint8 type'
        assert intensity_im.ndim==2, 'Not intensity image'
        # creating mask of overexposed area
       
        overexpo_mask=np.empty(intensity_im.shape, dtype='bool') 
        overexpo_mask=intensity_im==255
        overexpo_mask=255*overexpo_mask.astype(dtype=np.uint8) 
        return overexpo_mask
    
    def writeDiagnostics(self, savedir=None):
        if savedir is None:
            savedir=os.path.dirname(self.image_file)
        head, tail = str.split(os.path.basename(self.image_file),'.')
        diag_image_file=os.path.join(savedir,head+'_diagnostics.csv')
        out = open(os.path.join(diag_image_file), 'wt')
        w = csv.DictWriter(out, delimiter=';', fieldnames=['measures','values'])
        w.writeheader()
        for key, value in self.measures.items():
            w.writerow({'measures' : key, 'values' : value})
        out.close()
        
    def saveDiagImage(self, im, diag_id, savedir=None):
        if savedir is None:
            savedir=os.path.dirname(self.image_file)
        head, tail = str.split(os.path.basename(self.image_file),'.')
        diag_image_file=os.path.join(savedir,head+diag_id+'.jpg')
        io.imsave(diag_image_file,im)


def illumination_inhomogenity(csp, bg_mask, vis_diag):
    # using inpainting techniques
    assert csp.dtype=='uint8', 'Not uint8 type'
    
    gray=csp[:,:,2].copy()  
    
    gray[bg_mask==0]=0
    gray_s, scale=imtools.imRescaleMaxDim(gray, 64, interpolation = 0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask=img_as_ubyte(gray_s==0)
    inpainted =  inpaint.inpaint_biharmonic(gray_s, mask, multichannel=False)
    inpainted = gaussian(inpainted, 15, mode='reflect')
    if vis_diag:
        fi=plt.figure('inhomogen illumination')
        axi=fi.add_subplot(111)
        divider = make_axes_locatable(axi)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        i=axi.imshow(inpainted,cmap='jet')
        fi.colorbar(i, cax=cax, orientation='vertical')
        plt.show()  
    csp_corrected=img_as_float(csp)
    with warnings.catch_warnings():
        csp_corrected[:,:,2]=csp_corrected[:,:,2]+1-resize(inpainted, (gray.shape), order = 1)
        csp_corrected[csp_corrected>1]=1
        csp_corrected=img_as_ubyte(csp_corrected)
    #csp_corrected[:,:,2]=imtools.normalize(csp_corrected[:,:,2],vis_diag=False)
    return csp_corrected

