# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:13:50 2017

@author: SzMike
"""

import warnings
import csv
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.transform import resize
from skimage import feature
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
        self.param=cfg.param()
        self.vis_diag=vis_diag
        self.image_file=image_file
        self.image_shape=(im.shape[0],im.shape[1])
        self.do_illimination_corection=False
        self.do_blob_detection=False
        self.nRBC=0
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hsv = img_as_ubyte(color.rgb2hsv(im))
            l_dim=2 # luminosity dimension in hsv           
            self.hsv_small, scale = imtools.imRescaleMaxDim(self.hsv, self.param.small_size, interpolation=0)            
        
        intensity_im=self.hsv[:,:,l_dim]            
        # Calculate overexposition mask
        self.mask_over=self.overMask(intensity_im)            
        # Estimate inhomogen illumination
        self.cent_init, \
        self.bckg_pct, \
        bckg_inhomogenity_pct, \
        self.hsv_corrected, \
        self.im_corrected=self.illumination_correction(do=self.do_illimination_corection)
        
        if self.mask_over.sum()>0 and vis_diag:
            imtools.maskOverlay(im,self.mask_over,0.5,vis_diag=self.vis_diag,fig='overexposition mask')
        
        self.measures={}
        self.imhists=imtools.colorHist(im,mask=255-self.mask_over,vis_diag=False,fig='rgb')
        self.hsvhists=imtools.colorHist(self.hsv,mask=255-self.mask_over,vis_diag=False,fig='hsv')
        
        self.cumh_rgb, self.siqr_rgb = self.semi_IQR(self.imhists) # Semi-Interquartile Range
        self.cumh_hsv, self.siqr_hsv = self.semi_IQR(self.hsvhists) # Semi-Interquartile Range

        cumh=self.cumh_hsv[1] # saturation
        self.sat_q95=np.argwhere(cumh>0.95)[0,0]
        self.sat_q05=np.argwhere(cumh>0.05)[0,0]
        #self.sat_peak=np.argmax(self.hsvhists[1])
        
        self.ch_maxvar=np.argmax(self.siqr_rgb)

# TODO: allow adaptive setting
        self.h_min_wbc=255*self.param.wbc_range_in_hue[0]
        self.h_max_wbc=255*self.param.wbc_range_in_hue[1]

        minI=np.argwhere(self.cumh_hsv[l_dim]>0.05)[0,0]
        maxI=np.argwhere(self.cumh_hsv[l_dim]>0.95)[0,0]
        
         # Estimate RBC radius
        if self.do_blob_detection:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # TODO: use rgb cahnnel with max siqr, adjust threshold based on siqr
                gray, scale=imtools.imRescaleMaxDim(self.im_corrected[:,:,self.ch_maxvar],\
                                                     self.param.middle_size, interpolation=0)
                self.param.rbcR=self.blob_detection(255-gray,scale=scale,max_res=150,min_res=10,\
                                            threshold=0.5*self.siqr_rgb[self.ch_maxvar]/255, vis_diag=vis_diag)   
        
        # fill up initial measures
                                                                          
        self.measures['siqr_rgb']=self.siqr_rgb
        self.measures['siqr_hsv']=self.siqr_hsv
        self.measures['ch_maxvar']=self.ch_maxvar # channel with maximal variability
        self.measures['maxI']=maxI.astype('float64')
        self.measures['minI']=minI.astype('float64')
        self.measures['contrast']=(self.measures['maxI']-self.measures['minI'])/(self.measures['maxI']+self.measures['minI'])
        self.measures['overexpo_pct']=(self.mask_over>0).sum()/self.mask_over.size
        self.measures['global_entropy']=np.NaN # global homogenity
        self.measures['global_var']=np.NaN # global variance
        self.measures['saturation_q95']=self.sat_q95
        self.measures['saturation_q05']=self.sat_q05
        self.measures['bckg_inhomogenity_pct']=bckg_inhomogenity_pct
        self.measures['bckg_pct']=self.bckg_pct
        self.measures['rbcR']=self.param.rbcR
        
        self.error_list=[]
        #self.checks()

    def checks(self):
        self.error_list=[]
#        if np.logical_not(self.measures['ch_maxvar']==1):
#            self.error_list.append('ch_maxvar')
        if self.measures['contrast']<0.25:
            self.error_list.append('contrast')
        if self.measures['overexpo_pct']>0.1:
            self.error_list.append('overexpo_pct')
        if self.measures['saturation_q95']<80:
            self.error_list.append('saturation_q95')
        if self.measures['saturation_q05']>30:
            self.error_list.append('saturation_q05')
        if self.measures['bckg_pct']>0.75:
            self.error_list.append('bckg_pct')
        if self.measures['rbcR']<15 or self.measures['rbcR']>40:
            self.error_list.append('rbcR')
        if (len(self.error_list)>0):
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
    
    def illumination_correction(self,do=False):
        cent_init, label_mask = segmentations.segment_hsv(self.hsv_small, chs=(1,1,2),\
                                                          n_clusters=5,\
                                                          vis_diag=self.vis_diag)
        ind_val=np.argsort(cent_init[:,2]) # sure background - highest intensity
        # TODO: check ind_val[-2]
        mask_bg_sure=label_mask == ind_val[-1]
        bckg_pct=(mask_bg_sure>0).sum()/mask_bg_sure.size
        mask_bg_sure=morphology.binary_erosion(mask_bg_sure,morphology.disk(2));
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_bg_sure= img_as_ubyte(resize(mask_bg_sure,self.image_shape,order=0))
# TODO: check background distance transform and coverage (area) - should not be too large, too small
        if do:                                                 
            bckg_inhomogenity_pct, hsv_corrected=illumination_inhomogenity_hsv(self.hsv, mask_bg_sure, vis_diag=self.vis_diag)
        else:
            bckg_inhomogenity_pct=0
            hsv_corrected=self.hsv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im_corrected=img_as_ubyte(color.hsv2rgb(hsv_corrected))
        if self.vis_diag:
            f=plt.figure('intensity corrected images')
            ax1=f.add_subplot(121)
            ax1.imshow(im_corrected)
            ax2=f.add_subplot(122)
            ax2.imshow(hsv_corrected)
        return cent_init, bckg_pct, bckg_inhomogenity_pct, hsv_corrected, im_corrected
    
    def blob_detection(self,gray,scale=1,threshold=0.1, max_res=100,min_res=10,vis_diag=False):
        
        blobs = feature.blob_log(gray, min_sigma=max(gray.shape)/max_res, max_sigma=max(gray.shape)/min_res,\
                                 num_sigma=20, threshold=threshold, overlap=1)
         #    blobs = feature.blob_dog(gray, max_sigma=20, threshold=.1)
         #    blobs = feature.blob_doh(gray, max_sigma=30, threshold=.005)

        # Compute radii in the 3rd column.
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    
        hist_r,bin_edges=np.histogram(blobs[:,2]/scale,20)
        hist_r=hist_r[1:-1]
        bin_edges=bin_edges[1:-1]
        rMAX=((bin_edges[np.argmax(hist_r)]+bin_edges[np.argmax(hist_r)+1])/2)

        if vis_diag:
            fi=plt.figure('blobs')
            fi.clear()
            ax1=fi.add_subplot(121)
            ax1.imshow(gray,cmap='gray')
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='g', linewidth=2, fill=False)
                ax1.add_patch(c)
                ax1.set_axis_off()
            ax2=fi.add_subplot(122)
            ax2.bar(bin_edges[:-1], hist_r, width = 1)
            plt.tight_layout()
            plt.show()
    
        return rMAX
    

    def writeDiagnostics(self, savedir=None):
        if savedir is None:
            savedir=os.path.dirname(self.image_file)
        head, tail = str.split(os.path.basename(self.image_file),'.')
        diag_image_file=os.path.join(savedir,head+'_diagnostics.csv')
        with open(os.path.join(diag_image_file), 'wt',newline='') as f:
            w = csv.DictWriter(f, delimiter=';', fieldnames=['measures','values'])
            w.writeheader()
            for key, value in self.measures.items():
                w.writerow({'measures' : key, 'values' : value})
        
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
        hsv_corrected=img_as_float(hsv)
        hsv_corrected[:,:,2]=hsv_corrected[:,:,2]+1-resize(inpainted, (gray.shape), order = 1)
        hsv_corrected[hsv_corrected>1]=1
        hsv_corrected=img_as_ubyte(hsv_corrected)
    #hsv_corrected[:,:,2]=imtools.normalize(hsv_corrected[:,:,2],vis_diag=False)
    return bckg_inhomogenity_pct, hsv_corrected

