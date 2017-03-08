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


from skimage import measure
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

# %matplotlib qt5
 
import cfg
import imtools
import diagnostics
import segmentations
import cell_morphology
import annotations

%matplotlib qt5

image_dir=r'e:\WBC\data\Test\WBC Types\Problem'
included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)


image_file=image_list_indir[5]

vis_diag=False

for image_file in image_list_indir:

    print(image_file)
    

    
    # READ THE IMAGE
    im = io.imread(image_file) # read uint8 image
   
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
    
    output_dir=diag.param.getOutDir(dir_name='output')
    diag_dir=diag.param.getOutDir(dir_name='diag')
            
                                   
    # SMOOTHING
    #im_smooth=imtools.smooth3ch(im,r=3)
    #hsv_smooth=imtools.smooth3ch(diag.hsv_corrected,r=3)       
  
    """
    Foreground masks
    """  
    
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,diag.param.middle_size,interpolation = 0)
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,diag.param.middle_size,interpolation = 0)
 
    # create foreground mask using previously set init centers
    clust_centers_0, label_0 = segmentations.segment_hsv(hsv_resize, init_centers=diag.cent_init,\
                                                         chs=(1,1,2),\
                                                         n_clusters=4,\
                                                         vis_diag=vis_diag)   
    label_fg_bg=cell_morphology.rbc_labels(im,clust_centers_0,label_0)


    """
    RESIZE MASKS TO ORIGINAL
    """
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")    
#        label_fg_bg_orig = img_as_ubyte(resize(label_fg_bg,diag.image_shape, order = 0))
#        #label_wbc_orig = img_as_ubyte(resize(label_wbc,diag.image_shape, order = 0))
#        #label_fg_bg_orig[label_wbc_orig>0]=0
    
    """
    RBC detection
    """
   
#    mask_fg_clear=cell_morphology.rbc_mask_morphology(im_resize,label_fg_bg,diag.param,scale=scale,label_tsh=30,vis_diag=vis_diag,fig='31')    
#      
#    markers_rbc=cell_morphology.rbc_markers_from_mask(mask_fg_clear,diag.param,scale=scale)
#    segmentation.clear_border(markers_rbc,buffer_size=int(50*scale),in_place=True)


    markers_rbc_2, rbcR=cell_morphology.blob_markers(label_fg_bg>30,diag.param,rbc=True,scale=scale,fill_tsh=0.75,
                                                     vis_diag=vis_diag,fig='31')
    diag.param.rbcR=rbcR
    segmentation.clear_border(markers_rbc_2,buffer_size=int(50*scale),in_place=True)

    rbc_2=imtools.overlayImage(im_resize,markers_rbc_2>0,(1,0,0),1,vis_diag=vis_diag,fig='rbc_mask_2')   
#    rbc_1=imtools.overlayImage(im_resize,markers_rbc>0,(1,0,0),1,vis_diag=vis_diag,fig='rbc_mask_1')   
   
    #diag.saveDiagImage(rbc_2,'rbc_mask_2',savedir=diag_dir)
#    diag.saveDiagImage(rbc_1,'rbc_mask_1',savedir=diag_dir)

    """
    WBC masks
    """
    
    mask_sat=np.logical_and(np.logical_and(hsv_resize[:,:,0]>0,hsv_resize[:,:,0]<300),\
                                           hsv_resize[:,:,1]>diag.sat_q90)
    
    #mask_wbc=morphology.binary_opening(mask_wbc,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR)))
    wbc_nuc=imtools.overlayImage(im_resize,mask_sat,(0,1,1),1,vis_diag=vis_diag,fig='nuc_mask')
   
    #diag.saveDiagImage(wbc_nuc,'nuc_mask_1',savedir=diag_dir)
    
    clust_centers_1, label_1 = segmentations.segment_hsv(hsv_resize, mask=mask_sat,\
                                                    cut_channel=1, chs=(0,0,0),\
                                                    n_clusters=4,\
                                                    vis_diag=vis_diag) 
    # find cluster with highest saturation
    print(diag.param.rbcR)
    clust_hue=clust_centers_1[:,0]
    
    clust_sat=np.zeros(len(clust_hue))    
    mask_wbc=np.zeros(label_1.shape)
    label_wbc=np.zeros(label_1.shape)
    for i in range(clust_hue.shape[0]):
        hist_hsv=imtools.colorHist(hsv_resize,mask=label_1==i)
        cumh_hsv, siqr_hsv = diag.semi_IQR(hist_hsv) # Semi-Interquartile Range
        clust_sat[i]=np.argwhere(cumh_hsv[1]>0.99)[0,0]
    for i in range(clust_hue.shape[0]):
        if clust_sat[i]>(clust_sat.max()+diag.sat_q90)/2:
            mask_wbc[label_1==i]=1
            mask_temp=label_1==i
#            mask_temp=morphology.binary_opening(mask_temp,morphology.disk(np.ceil(scale*diag.param.cell_bound_pct*diag.param.rbcR)))            
#            mask_temp=morphology.binary_closing(mask_temp,morphology.disk(np.ceil(0.75*scale*diag.param.rbcR)))            
            label_wbc[mask_temp]=1

# TODO use regionprops on mask size
    
    #diag.param.cell_bound_pct=0.2
    #mask_wbc=morphology.binary_opening(mask_wbc,morphology.disk(int(scale*diag.param.cell_bound_pct*param.rbcR)))
    wbc_nuc_2=imtools.overlayImage(im_resize,mask_wbc>0,(1,1,0),0.5,vis_diag=False,fig='nuc_mask_2')   
    wbc_nuc_2=imtools.overlayImage(wbc_nuc_2,label_wbc>0,(0,1,0),1,vis_diag=vis_diag,fig='nuc_mask_2')
#   
#    diag.saveDiagImage(wbc_nuc_2,'nuc_mask_2',savedir=diag_dir)

    markers_wbc_2=cell_morphology.wbc_markers(label_wbc>0,diag.param,scale=scale,fill_tsh=0.25,vis_diag=vis_diag,fig='32')

    wbc=imtools.overlayImage(rbc_2,markers_wbc_2>0,(0,1,0),1,vis_diag=vis_diag,fig='rbc_mask_2')   
#    rbc_1=imtools.overlayImage(im_resize,markers_rbc>0,(1,0,0),1,vis_diag=vis_diag,fig='rbc_mask_1')   
   
    diag.saveDiagImage(wbc,'markers',savedir=diag_dir)
#    
"""
# create segmentation for WBC detection based on hue and saturation
    sat_min=max(np.sort(clust_centers_0[:,0])[0],30)
    #mask=np.logical_and(label_fg_bg>1,np.logical_and(hsv_resize[:,:,0]>diag.h_min_wbc,hsv_resize[:,:,0]<diag.h_max_wbc))
    mask_not_bckg=np.logical_and(label_fg_bg>1,hsv_resize[:,:,1]>sat_min)
       
    clust_centers_1, label_1 = segmentations.segment_hsv(hsv_resize, mask=mask_not_bckg,\
                                                    cut_channel=1, chs=(0,0,1),\
                                                    n_clusters=4,\
                                                    vis_diag=vis_diag) 
    clust_sat=clust_centers_1[:,2]
 
    mask_wbc_pot=cell_morphology.wbc_masks(label_1,clust_sat,scale,vis_diag=vis_diag)
    
    wbc_mask=imtools.overlayImage(im_resize,label_fg_bg==2,(0,1,1),1,vis_diag=False,fig='wbc_mask')
    wbc_mask=imtools.overlayImage(wbc_mask,label_fg_bg==32,(1,1,0),1,vis_diag=False,fig='wbc_mask')   
    wbc_mask=imtools.overlayImage(wbc_mask,mask_wbc_pot[0]>0,(0,1,0),1,vis_diag=vis_diag,fig='wbc_mask')

    diag.saveDiagImage(wbc_mask,'wbc_mask',savedir=diag_dir)

    
#    clust_sat=np.zeros(len(clust_centers_1[:,2]))
#    for i in range(len(clust_centers_1[:,2])):
#        mask_tmp=label_1==i
#        hist_hsv=imtools.colorHist(hsv_resize,mask=mask_tmp)
#        cumh_hsv, siqr_hsv = diag.semi_IQR(hist_hsv) # Semi-Interquartile Range
#        clust_sat[i]=np.argwhere(cumh_hsv[1]>0.99)[0,0]
#    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    
        label_fg_bg_orig = img_as_ubyte(resize(label_fg_bg,diag.image_shape, order = 0))
        mask_wbc_pot_orig = img_as_ubyte(resize(mask_wbc_pot[0]>0,diag.image_shape, order = 0))
   
    mask_fg=np.logical_or(label_fg_bg_orig>1,label_fg_bg_orig>30)
    mask_fg_open=morphology.binary_opening(mask_fg,morphology.star(2))
#   
#    mask_fg=label_mask==32
#    mask_fg_open_2=morphology.binary_closing(mask_fg,morphology.disk(1)).astype('uint8')
#    mask_fg=np.logical_or(mask_fg_open_1,mask_fg_open_2)
#   
#    mask_fg_filled=morphology.remove_small_holes(mask_fg_open>0, 
#                                                 min_size=param.cellFillAreaPct*param.rbcR*param.rbcR*np.pi, 
#                                                 connectivity=2)
#    mask_fg_clear=morphology.binary_opening(mask_fg_filled,morphology.disk(param.rbcR*param.cellOpeningPct)).astype('uint8')
#
#
#    skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
#    dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
#    # watershed seeds
#    # TODO - add parameters to cfg
#    local_maxi = feature.peak_local_max(dtf, indices=False, 
#                                        threshold_abs=0.5*param.rbcR,
#                                        footprint=np.ones((int(param.rbcR), int(param.rbcR))), 
#                                        labels=mask_fg_clear.copy())
#    markers, n_RBC = measure.label(local_maxi,return_num=True)
#
#    im_detect=imtools.overlayImage(im,morphology.binary_dilation(markers>0,morphology.disk(5)),\
#            (1,0,0),1,vis_diag=vis_diag,fig='detections')
#
#    labels_ws = morphology.watershed(-dtf, markers, mask=mask_fg_clear)
#    
#    pot_wbc=-np.ones(mask_wbc_pot_orig.shape)
#    j=0
#    for i in range(labels_ws.max()):
#        if i>0:
#            mask_tmp=labels_ws==i
#            if (mask_tmp).sum()<25*np.power(param.rbcR,2)*np.pi and\
#                np.logical_and(mask_tmp>0,mask_wbc_pot_orig>0).sum()>0.33*np.power(param.rbcR,2)*np.pi and\
#                np.logical_and(mask_tmp>0,mask_wbc_pot_orig>0).sum()/(mask_tmp).sum()>0.1:
#                pot_wbc[mask_tmp]=j
#                j=j+1
#   
#    im_wbc=imtools.overlayImage(im,pot_wbc>-1,\
#            (0,1,0),1,vis_diag=False,fig='wbc_detections')
#    im_detect=imtools.overlayImage(im_wbc,morphology.binary_dilation(markers>0,morphology.disk(5)),\
#            (1,0,0),1,vis_diag=vis_diag,fig='detections')   
#    
#    diag.saveDiagImage(im_detect,'detection',savedir=diag_dir)

#    hsv_smooth=imtools.smooth3ch(hsv_resize,r=3)   
#    hue_mask=color_mask.hue_mask(hsv_smooth,mask_wbc_pot,diag)
#    hue_mask=np.logical_and(hue_mask,mask_not_bckg)
#    im_hue=imtools.overlayImage(im_resize,hue_mask>0,(0,0,1),1,vis_diag=vis_diag,fig='wbc_hue_mask')
#    diag.saveDiagImage(im_hue,'hue_mask',savedir=diag_dir)

#    
#    gray_1=hsv_resize[:,:,0].copy()       
#    gray_2=im_resize[:,:,1].copy()       
#    #gray[label_fg_bg==1]=0    
#        
#    gg_1=imtools.getGradientMagnitude(gray_1)  
#    gg_2=imtools.getGradientMagnitude(gray_2)  
#    
#    gg=gg_1+gg_2
#    edgemap=gg>0.2
#    edgemap=morphology.closing(edgemap,morphology.disk(2))
#    
# 
#    im_m=imtools.overlayImage(im_resize,edgemap>0,(0,0,1),1,vis_diag=True,fig='wbc_mask')
#    
#    potmask=morphology.opening(mask_wbc_pot[0],morphology.disk(3))
#    im_m=imtools.overlayImage(im_m,potmask,(0,1,0),1,vis_diag=True,fig='wbc_mask')
#
#
#    cnts_WBC = measure.find_contours(potmask, 0.5)
#    
#    print(len(cnts_WBC))
#
#    for c in cnts_WBC:
#        if len(c)>5:
#              seed= ((np.mean(c,0))).astype('int64')
#              reg=region_growing.simple_region_growing(edgemap, seed, threshold=1, maxR=25)
#              im_m=imtools.overlayImage(im_m,reg>0,(0,1,0),1,vis_diag=vis_diag,fig='wbc_mask')
#    
#    im_m=imtools.overlayImage(im_m,potmask,(1,0,0),1,vis_diag=True,fig='wbc_mask')

#   
#
#    """
#    CREATE and SAVE DIAGNOSTICS IMAGES
#    """
#    
#    wbc_mask=imtools.overlayImage(im_resize,mask_wbc_pot[1]>0,(1,1,0),0.5,vis_diag=False,fig='wbc_mask')
#    wbc_mask=imtools.overlayImage(wbc_mask,mask_wbc_pot[0]>0,(0,1,0),1,vis_diag=False,fig='wbc_mask')
##    wbc_mask=imtools.overlayImage(wbc_mask,edgemap,(0,0,1),1,vis_diag=vis_diag,fig='wbc_mask')
#  
#    
#    im_wbc=imtools.overlayImage(im_resize,label_wbc==1,(0,1,1),1,vis_diag=False,fig='wbc')
#    im_wbc=imtools.overlayImage(im_wbc,label_wbc==2,(1,0,1),1,vis_diag=False,fig='wbc')
#    im_wbc=imtools.overlayImage(im_wbc,label_wbc==3,(1,1,0),1,vis_diag=False,fig='wbc')
#    im_wbc=imtools.overlayImage(im_wbc,label_wbc==4,(0,0,1),1,vis_diag=vis_diag,fig='wbc')  
#    
#    
#    diag.saveDiagImage(wbc_mask,'wbc_mask',savedir=diag_dir)
#    diag.saveDiagImage(im_wbc,'wbc_detect',savedir=diag_dir)
#    diag.saveDiagImage(im_m,'reggrow',savedir=diag_dir)
#    
#    
#    
##     from skimage.segmentation import active_contour
##
##    cnts_WBC = measure.find_contours(label_wbc==2, 0.5)
##    init=cnts_WBC[0]
##    
##    s = np.linspace(0, 2*np.pi, 100)
##    cent=np.mean(cnts_WBC[0],0)
##    x = cent[0] + 15*np.cos(s)
##    y = cent[1] + 15*np.sin(s)
##    init = np.array([x, y]).T
##    
##    snake = active_contour(im_resize,init, alpha=1, beta=10, gamma=0.001)
###    
##    fig = plt.figure('alma',figsize=(7, 7))
##    ax = fig.add_subplot(111)
##    plt.gray()
##    ax.imshow(im_resize)
##    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
##    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
##    ax.set_xticks([]), ax.set_yticks([])
##    ax.axis([0, im_resize.shape[1], im_resize.shape[0], 0])

"""