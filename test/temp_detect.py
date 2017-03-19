# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:46:42 2017

@author: SzMike
"""
import __init__


import os
import warnings
import time
import skimage.io as io
import numpy as np;
from skimage.transform import resize
from skimage import filters
import glob
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

import annotations
from scipy import ndimage
import time

%matplotlib qt5
plt.ioff()

#image_dir=r'e:\CELLDATA\Slides\1106_kezi_diapH_5_7_12'
image_dir=r'd:\Projects\WBC\data\Test'

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)


image_file=image_list_indir[2]

vis_diag=True

for image_file in image_list_indir:

    print(image_file)
    
    im = io.imread(image_file) # read uint8 image
    
    """
    DIAGNOSTICS
    """
                  
    start_time = time.time()
    
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
    
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
    WBC nucleus masks
    """   
    start_time = time.time()    
 
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=diag.sat_q95,scale=scale,vis_diag=vis_diag,fig='')

    print("--- %s seconds - WBC nucleus mask ---" % (time.time() - start_time))
    """
    CELL FOREGORUND MASK
    """
    start_time = time.time() 
    
    mask_cell=detections.cell_mask(hsv_resize,diag.param,scale=scale,mask=mask_nuc,init_centers=diag.cent_init,vis_diag=vis_diag,fig='cell_mask')
    
    print("--- %s seconds - CELL mask ---" % (time.time() - start_time))

    """
    CELL MARKERS AnD REGIONS
    """
    start_time = time.time()  
    
    markers_rbc, prop_rbc=detections.cell_markers_from_mask(mask_cell,diag.param,scale=scale,vis_diag=vis_diag,fig='cell_markers')         

    print("--- %s seconds - CELL markers ---" % (time.time() - start_time))
    """
    CREATE WBC REGIONS
    """
    start_time = time.time()      
    
    prop_wbc=detections.wbc_regions(mask_nuc,diag.param,scale=scale,vis_diag=True)
    
    print("--- %s seconds - CELL markers ---" % (time.time() - start_time))
  
    """
    CREATE RBC REGIONS
    """    
    
    diag.measures['n_WBC']=len(prop_wbc)
    diag.measures['n_RBC']=len(prop_rbc)
    
    im_rbc=imtools.overlayImage(im_resize,mask_cell,(1,0,0),0.5,vis_diag=False,fig='rbc')    
    im_nuc=imtools.overlayImage(im_rbc,mask_nuc,(0,0,1),0.5,vis_diag=vis_diag,fig='nuc')    
    
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
    CREATE shapes
    """
 
    shapelist_WBC=[]
    for p in prop_wbc:
        # centroid is in row,col
         pts=[(p.centroid[1]/scale+0.75*p.major_axis_length*np.cos(theta*2*np.pi/20)/scale,p.centroid[0]/scale+0.75*p.major_axis_length*np.sin(theta*2*np.pi/20)/scale) for theta in range(20)] 
         #pts=[(p.centroid[1]/scale,p.centroid[0]/scale)]
         one_shape=('WBC','circle',pts,'None','None')
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
    # Plot manual
    fig = plt.figure('detections',figsize=(20,20))
    fig=imtools.plotShapes(im,shapelist,color='r',\
                                   detect_shapes='RBC',text=(''),fig=fig)
    fig=imtools.plotShapes(im,shapelist,color='b',\
                                   detect_shapes='WBC',text=('WBC'),fig=fig)

    diag.saveDiagFigure(fig,'detections',savedir=diag_dir)
    plt.close(fig) 