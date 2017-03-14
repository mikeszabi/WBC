# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:46:42 2017

@author: SzMike
"""
import __init__

from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
import os
import warnings
import skimage.io as io
import numpy as np;
from skimage.transform import resize
from skimage import filters
import glob
from skimage import color


from skimage import measure
from skimage import draw

from skimage import img_as_ubyte
from matplotlib import pyplot as plt

# %matplotlib qt5
 
import cfg
import imtools
import diagnostics
import segmentations
import detections

import cell_morphology
import annotations
from scipy import ndimage

import cv2

%matplotlib qt5

#image_dir=r'e:\CELLDATA\Slides\1106_kezi_diapH_5_7_12'
image_dir=r'd:\Projects\WBC\data\Test\WBC Types\Problem'

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)


image_file=image_list_indir[0]

vis_diag=False

for image_file in image_list_indir:

    print(image_file)
    i=0
    
    im = io.imread(image_file) # read uint8 image
   
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
    
    output_dir=diag.param.getOutDir(dir_name=os.path.join('output'))
    diag_dir=diag.param.getOutDir(dir_name=os.path.join('diag'))
   
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,diag.param.middle_size,interpolation = 0)
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,diag.param.middle_size,interpolation = 0)
 
    """
    WBC nucleus masks
    """
# create segmentation for WBC detection based on hue and saturation
#    label_wbc=np.logical_and(np.logical_and(hsv_resize[:,:,0]>diag.param.wbc_range_in_hue[0]*255,\
#                                            hsv_resize[:,:,0]<diag.param.wbc_range_in_hue[1]*255),\
#                                            hsv_resize[:,:,1]>diag.sat_q95)
#
#    
##    im_wbc=imtools.overlayImage(im_resize,label_fg_bg==2,(1,0,1),0.6,vis_diag=vis_diag,fig='sat')    
##    im_wbc=imtools.overlayImage(im_wbc,label_fg_bg==3,(1,1,0),0.3,vis_diag=vis_diag,fig='sat')    
#
#    im_wbc=imtools.overlayImage(im_resize,label_wbc>0,(0,1,1),0.5,vis_diag=vis_diag,fig='sat')    
#    mask_fg=label_wbc>0
#    mask_fg=morphology.binary_opening(mask_fg,morphology.disk(2))
    #mask_fg=morphology.binary_closing(mask_fg,morphology.disk(2))
    
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=diag.sat_q95,scale=scale,vis_diag=False,fig='')
    im_wbc=imtools.overlayImage(im_resize,mask_nuc,(0,0,1),0.8,vis_diag=vis_diag,fig='sat')    

    label_img = measure.label(mask_nuc, connectivity=mask_nuc.ndim)
    props = measure.regionprops(label_img)
    props_large=props.copy()
    
    for p in props:
        if p.area<0.33*diag.param.rbcR**2*np.pi*scale**2:
            props_large.remove(p)
    
#    fig = plt.figure('snake')
#    ax = fig.add_subplot(111)
#    ax.imshow(im_wbc)
    
    radius = 4
    n_points = 4 * radius
    lbp=feature.local_binary_pattern(hsv_resize[:,:,2], n_points,radius, method='uniform')
    
    for p in props_large:
    #    init = draw.polygon([p.bbox[1], p.bbox[3], p.bbox[3], p.bbox[1]],
    #                        [p.bbox[0], p.bbox[0], p.bbox[2], p.bbox[2]])
        crop_image=hsv_resize[p.bbox[0]:p.bbox[2],p.bbox[1]:p.bbox[3],:]
        init = draw.circle(int(p.centroid[0]), int(p.centroid[1]), int(p.minor_axis_length))
        crop_image=im_resize[int(p.centroid[0])-int(p.major_axis_length):int(p.centroid[0])+int(p.major_axis_length),\
                            int(p.centroid[1])-int(p.major_axis_length):int(p.centroid[1])+int(p.major_axis_length),:]
# major_axis_length - make sure the whole detected nucleid is covered
#        ax.plot(init[1], init[0], '.', lw=1)
        
        h_pixs=hsv_resize[init[0],init[1],0]
        l_pixs=lbp[init[0],init[1]]
        s_pixs=hsv_resize[init[0],init[1],1]
       
        hp,beh=np.histogram(h_pixs,bins=64,range=(0,256),normed=True)
        sp,bes=np.histogram(s_pixs,bins=64,range=(0,128),normed=True)
        lp,bel=np.histogram(l_pixs,bins=n_points+2,range=(0,n_points+2),normed=True)
        hist, xedges, yedges = np.histogram2d(h_pixs, s_pixs, (np.linspace(0, 256, 4), np.linspace(0, 256, 8)))
        
        fig2 = plt.figure('wbc')
        ax21= fig2.add_subplot(311)
        ax21.imshow(crop_image)
        ax20= fig2.add_subplot(312)
        xidx = np.clip(np.digitize(h_pixs, xedges), 0, hist.shape[0]-1)
        yidx = np.clip(np.digitize(s_pixs, yedges), 0, hist.shape[1]-1)
        c = hist[xidx, yidx]
        ax20.scatter(h_pixs, s_pixs, c=c)
        ax20.set_xlim([0, 255])
        ax20.set_ylim([0, 255])
        
#        ax21 = fig2.add_subplot(411)
#        
#        
#        
#        ax21.imshow(crop_image)
#        ax22 = fig2.add_subplot(412)
#        ax22.plot(beh[1:],hp)
#        ax22 = fig2.add_subplot(413)
#        ax22.plot(bes[1:],sp)
        ax22 = fig2.add_subplot(313)
        ax22.plot(bel[1:],lp)
        i=i+1
        diag.saveDiagFigure(fig2,'wbc_detections_'+str(i),savedir=diag_dir)
        plt.close(fig2)    

    """
    RBC detection
    """
    
#    diag.saveDiagFigure(fig,'wbc_detections',savedir=diag_dir)
#    plt.close(fig)    

#    max_dim=max(mask_fg.shape)
#    
#    min_r=2*int(max(mask_fg.shape)/scale/100/2)
#    max_r=2*(max(mask_fg.shape)/scale/20/2)
#    r_list = np.linspace(start=min_r, stop=max_r, num=(max_r-min_r)/2+1)
#       
#       
#    start_r=0
#    r_list=r_list[r_list>start_r]
#          
#    mask = (mask_fg).astype('float64')
#    
#    im_filtered = [ndimage.convolve(mask, morphology.disk(r*scale))/(morphology.disk(r*scale).sum()) for r in r_list]
#    
#    fill_cube = np.dstack(im_filtered)
#    
#    fp=int(max_dim/scale/50)   
#    threshold=fill_tsh
#    local_maxima_fill = feature.peak_local_max(fill_cube, 
#                                          threshold_abs=threshold,
#                                          indices=True,
#                                          footprint=np.ones((fp,fp,3)),
#                                          threshold_rel=0.0,
#                                          exclude_border=False)
#    import blobtools
#    local_maxima_fill_prune=blobtools.prune_blobs(local_maxima_fill, 0.5)
#    
#    if vis_diag:
#        fig=plt.figure('circle image')
#        axs=fig.add_subplot(111)
#        axs.imshow(color.gray2rgb(255*mask).astype('uint8'))  
#        for l in local_maxima_fill_prune:
#            circ=plt.Circle((l[1],l[0]), radius=r_list[l[2]]*scale, color='g', fill=False)
#            axs.add_patch(circ)
#            
#    markers_r=np.zeros(mask_fg.shape)    
#    for l in local_maxima_fill:
#        markers_r[l[0],l[1]]=r_list[l[2]]    
#    # TODO: get rid of dilatation
#    markers=morphology.binary_dilation(markers_r,morphology.disk(1)).astype('uint8')
#    
#    markers_wbc_nuc=markers
#    #markers_wbc_nuc=cell_morphology.wbc_markers(label_wbc>0,diag.param,scale=scale,fill_tsh=0.33,vis_diag=vis_diag,fig='wbc_nuc')
#    #segmentation.clear_border(markers_wbc_nuc,buffer_size=diag.param.middle_border,in_place=True)
#  
#    im_wbc=imtools.overlayImage(im_resize,markers_wbc_nuc,(0,1,0),1,vis_diag=vis_diag,fig='wbc')    
#    
##    diag.saveDiagImage(im_detect,'detections',savedir=diag_dir)