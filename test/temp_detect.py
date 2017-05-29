# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:46:42 2017

@author: SzMike
"""

import os
import glob
import warnings
import time
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL

import numpy as np
from scipy import ndimage
from skimage.transform import resize
from skimage import filters
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
from skimage import color
from skimage import measure
from skimage import draw
from skimage import img_as_ubyte
from scipy import ndimage
from matplotlib import pyplot as plt
from matplotlib.path import Path

# %matplotlib qt5
 
import cfg
import imtools
import diagnostics
import segmentations
import detections
import classifications
import annotations

import time

#image_dir=r'c:\Users\SzMike\OneDrive\WBC\DATA\Annotated\9426 Javított E'
image_dir=os.path.join(os.path.curdir,'data','Test')

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)


image_file=image_list_indir[22]

vis_diag=True
if vis_diag==True:
    plt.ion()
    %matplotlib qt5
else:
    plt.ioff()

cnn=classifications.cnn_classification()

for image_file in image_list_indir:

    print(image_file)
    
    im = io.imread(image_file) # read uint8 image
    
    """
    DIAGNOSTICS
    """
                  
    start_time = time.time()
    
    diag=diagnostics.diagnostics(im,image_file,vis_diag=vis_diag)
    
    print('--- %s seconds - Diagnostics ---' % (time.time() - start_time))
    
    """
    CREATE OUTPUT AND DIAG DIRS
    """
       
    output_dir=diag.param.getOutDir(dir_name=os.path.join('output'))
    diag_dir=diag.param.getOutDir(dir_name=os.path.join('diag'))
    
    """
    RESIZE
    """    
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,diag.param.middle_size,interpolation = 0)
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,diag.param.middle_size,interpolation = 0)
    
    """
    WBC detection
    """   
    start_time = time.time()    
 
    sat_tsh=max(diag.sat_q95,diag.param.wbc_min_sat)
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=sat_tsh,scale=scale,vis_diag=vis_diag,fig='')

    print("--- %s seconds - WBC nucleus mask ---" % (time.time() - start_time))
    
    #CREATE WBC REGIONS
    
    start_time = time.time()      
    
    prop_wbc=detections.wbc_regions(mask_nuc,diag.param,scale=scale,vis_diag=True)
    
    print("--- %s seconds - CELL markers ---" % (time.time() - start_time))
   
    """
    CELL DETECTION
    """
    # CELL FOREGORUND MASK
    start_time = time.time() 
    
    mask_cell=detections.cell_mask(hsv_resize,diag.param,scale=scale,mask=mask_nuc,init_centers=diag.cent_init,vis_diag=vis_diag,fig='cell_mask')
    
    print("--- %s seconds - CELL mask ---" % (time.time() - start_time))

    # RBC MARKERS and REGIONS
    start_time = time.time()  
    
    markers_rbc, prop_rbc=detections.cell_markers_from_mask(mask_cell,diag.param,scale=scale,vis_diag=vis_diag,fig='cell_markers')         

    print("--- %s seconds - CELL markers ---" % (time.time() - start_time))
    
    """
    COUNTING
    """
    diag.measures['n_WBC']=len(prop_wbc)
    diag.measures['n_RBC']=len(prop_rbc)
    
    im_rbc=imtools.overlayImage(im_resize,mask_cell,(1,0,0),0.5,vis_diag=False,fig='rbc')    
    im_nuc=imtools.overlayImage(im_rbc,mask_nuc,(0,0,1),0.5,vis_diag=vis_diag,fig='nuc+rbc')    
    
#    fig = plt.figure('detections')
#    ax = fig.add_subplot(111)
#    ax.imshow(im_resize)    
#    for p in prop_wbc:
#        cnt = draw.circle_perimeter(int(p.centroid[0]), int(p.centroid[1]), int(p.minor_axis_length))
#        cnt=(cnt[1][np.linspace(0,len(cnt[1])-1,21,dtype='uint8',endpoint=False)],cnt[0][np.linspace(0,len(cnt[0])-1,21,dtype='uint8',endpoint=False)]) # row,col -> x,y
#        ax.plot(cnt[0], cnt[1], '.', lw=1)
   
#    diag.saveDiagFigure(fig,'wbc_detections',savedir=diag_dir)
#    plt.close(fig) 

    """
    PARAMETERS for WBC NORMALIZATION 
    """
    if mask_nuc.sum()>0:
        pixs=im_resize[mask_nuc,]
        diag.measures['nucleus_median_rgb']=np.median(pixs,axis=0)        

    
    """
    CREATE shapes AND classify WBC 
    """
 
    shapelist_WBC=[]
    for p in prop_wbc:
        # centroid is in row,col
         pts=[(p.centroid[1]/scale+0.8*p.major_axis_length*np.cos(theta*2*np.pi/20)/scale,p.centroid[0]/scale+0.8*p.major_axis_length*np.sin(theta*2*np.pi/20)/scale) for theta in range(20)] 

         #pts=[(p.centroid[1]/scale,p.centroid[0]/scale)]
         one_shape=('None','circle',pts,'None','None')
         
         # WBC classification
         
#         if min((im.shape[1],im.shape[0])-np.max(one_shape[2],axis=0))<0\
#                or min(np.min(one_shape[2],axis=0))<0:
#            continue
         
         im_cropped,o,r=imtools.crop_shape(im,one_shape,\
                                            diag.param.rgb_norm,diag.measures['nucleus_median_rgb'],\
                                            scale=scale,adjust=True)
         if im_cropped is not None and cnn is not None:
             # do the actual classification
             wbc_label, pct=cnn.classify(im_cropped)
             # redefiniton of wbc type
             one_shape=(wbc_label[0],'circle',pts,'None','None')
             
         shapelist_WBC.append(one_shape)
         
    shapelist_RBC=[]
    for p in prop_rbc:
        pts=[(p.centroid[1]/scale,p.centroid[0]/scale)]
        is_in_wbc=False
        for shape in shapelist_WBC:
             bb=Path(shape[2])
             intersect = bb.contains_points(pts)    
             if intersect.sum()>0:
                 is_in_wbc=True
        if not is_in_wbc:
            one_shape=('RBC','circle',pts,'None','None')
            shapelist_RBC.append(one_shape)
    shapelist=shapelist_WBC
    shapelist.extend(shapelist_RBC)
    
    """
    REMOVE ANNOTATIONS CLOSE TO BORDER
    """
    
    shapelist=annotations.remove_border_annotations(shapelist,im.shape,diag.param.border)
   
    
    # Plot manual
    fig = plt.figure('detections',figsize=(20,20))
    fig=imtools.plotShapes(im,shapelist,color='r',\
                                   detect_shapes='RBC',text=(''),fig=fig)
    fig=imtools.plotShapes(im,shapelist,color='b',\
                                   detect_shapes=list(diag.param.wbc_basic_types.keys()),\
                                                     text=list(diag.param.wbc_basic_types.keys()),fig=fig)

    diag.saveDiagFigure(fig,'detections',savedir=diag_dir)
    plt.close(fig) 
