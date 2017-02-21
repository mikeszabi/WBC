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
from skimage import morphology
from skimage import color
import skimage.io as io
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imtools
import cfg
import segmentations


class diagnostics:
    def __init__(self,im,image_file,vis_diag=False,write=False):
        assert im.ndim==3, 'Not 3 channel image'
        assert im.dtype=='uint8', 'Not byte image'     
        param=cfg.param()
        self.vis_diag=vis_diag
        self.image_file=image_file
        self.image_shape=(im.shape[0],im.shape[1])
        
        # TODO: add blob detection, estimate RBC size, statistics on blob localization
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hsv = img_as_ubyte(color.rgb2hsv(im))
            l_dim=2 # luminosity dimension in hsv           
            self.hsv_small, scale = imtools.imRescaleMaxDim(self.hsv, param.small_size, interpolation=0)            
            self.intensity_im=self.hsv[:,:,l_dim]            
            self.mask_over=self.overMask(self.intensity_im)            
            self.cent_init, \
            bckg_inhomogenity_pct, \
            self.hsv_corrected, \
            self.im_corrected=self.illumination_correction()
        
        if self.mask_over.sum()>0:
            imtools.maskOverlay(im,self.mask_over,0.5,vis_diag=self.vis_diag,fig='overexposition mask')
        
        self.measures={}
        self.imhists=imtools.colorHist(im,mask=255-self.mask_over,vis_diag=vis_diag,fig='rgb')
        self.hsvhists=imtools.colorHist(self.hsv,mask=255-self.mask_over,vis_diag=vis_diag,fig='hsv')
        
        self.cumh_rgb, self.siqr_rgb = self.semi_IQR(self.imhists) # Semi-Interquartile Range
        self.cumh_hsv, self.siqr_hsv = self.semi_IQR(self.hsvhists) # Semi-Interquartile Range

        cumh=self.cumh_hsv[1] # saturation
        self.sat_q90=np.argwhere(cumh>0.9)[0,0]
        self.sat_q10=np.argwhere(cumh>0.1)[0,0]
        #self.sat_peak=np.argmax(self.hsvhists[1])

# TODO: allow adaptive setting
        self.h_min_wbc=255*param.wbc_range_in_hue[0]
        self.h_max_wbc=255*param.wbc_range_in_hue[1]

        minI=np.argwhere(self.cumh_hsv[l_dim]>0.05)[0,0]
        maxI=np.argwhere(self.cumh_hsv[l_dim]>0.95)[0,0]
                                                                          
        self.measures['siqr_rgb']=self.siqr_rgb
        self.measures['siqr_hsv']=self.siqr_hsv
        self.measures['ch_maxvar']=np.argmax(self.siqr_rgb) # channel with maximal variability
        self.measures['maxI']=maxI.astype('float64')
        self.measures['minI']=minI.astype('float64')
        self.measures['contrast']=(self.measures['maxI']-self.measures['minI'])/(self.measures['maxI']+self.measures['minI'])
        self.measures['overexpo_pct']=(self.mask_over>0).sum()/self.mask_over.size
        self.measures['global_entropy']=np.NaN # global homogenity
        self.measures['global_var']=np.NaN # global variance
        self.measures['saturation_q90']=self.sat_q90
        self.measures['saturation_q10']=self.sat_q10
        self.measures['bckg_inhomogenity_pct']=bckg_inhomogenity_pct
        
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
        if self.measures['saturation_q90']<100:
            self.error_list.append('saturation_q90')
        if self.measures['saturation_q10']>30:
            self.error_list.append('saturation_q10')
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
    
    def illumination_correction(self):
        cent_init, label_mask = segmentations.segment_hsv(self.hsv_small, chs=(1,1,2),  n_clusters=4, vis_diag=self.vis_diag)
        ind_val=np.argsort(cent_init[:,2]) # sure background - highest intensity
        # TODO: check ind_val[-2]
        mask_bg_sure=morphology.binary_erosion(label_mask == ind_val[-1],morphology.disk(2));
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_bg_sure= img_as_ubyte(resize(mask_bg_sure,self.image_shape,order=0))
# TODO: check background distance transform and coverage (area) - should not be too large, too small

        bckg_inhomogenity_pct, hsv_corrected=illumination_inhomogenity_hsv(self.hsv, mask_bg_sure, vis_diag=self.vis_diag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im_corrected=img_as_ubyte(color.hsv2rgb(hsv_corrected))
            if self.vis_diag:
                f=plt.figure('intensity corrected image')
                ax=f.add_subplot(111)
                ax.imshow(im_corrected)
        return cent_init, bckg_inhomogenity_pct, hsv_corrected, im_corrected
    
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

    def saveDiagFigure(self, fig, diag_id, savedir=None):
        if savedir is None:
            savedir=os.path.dirname(self.image_file)
        head, tail = str.split(os.path.basename(self.image_file),'.')
        diag_image_file=os.path.join(savedir,head+diag_id+'.jpg')
        fig.savefig(diag_image_file)

def illumination_inhomogenity_hsv(hsv, mask_bg_sure, vis_diag=False):
    # using inpainting techniques
    assert hsv.dtype=='uint8', 'Not uint8 type'
    assert hsv.ndim==3, 'Not 3channel image'
    
    gray=hsv[:,:,2].copy()  
    
    gray[mask_bg_sure==0]=0
    gray_s, scale=imtools.imRescaleMaxDim(gray, 64, interpolation = 0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask=img_as_ubyte(gray_s==0)
    inpainted =  inpaint.inpaint_biharmonic(gray_s, mask, multichannel=False)
    inpainted = gaussian(inpainted, 15, mode='reflect')
    bckg_inhomogenity_pct=1-inpainted.min()/max(inpainted.max(),0)
    if vis_diag:
        fi=plt.figure('inhomogen illumination')
        axi=fi.add_subplot(111)
        divider = make_axes_locatable(axi)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        i=axi.imshow(inpainted,cmap='jet')
        fi.colorbar(i, cax=cax, orientation='vertical')
        plt.show()  
    with warnings.catch_warnings():
        hsv_corrected=img_as_float(hsv)
        hsv_corrected[:,:,2]=hsv_corrected[:,:,2]+1-resize(inpainted, (gray.shape), order = 1)
        hsv_corrected[hsv_corrected>1]=1
        hsv_corrected=img_as_ubyte(hsv_corrected)
    #hsv_corrected[:,:,2]=imtools.normalize(hsv_corrected[:,:,2],vis_diag=False)
    return bckg_inhomogenity_pct, hsv_corrected

