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
image_dir=r'd:\Projects\WBC\data\Test\WBC Types\Problem'

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)


image_file=image_list_indir[9]

vis_diag=False

for image_file in image_list_indir:

    print(image_file)
    i=0
    
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
 
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=diag.sat_q95,scale=scale,vis_diag=False,fig='')

    print("--- %s seconds - WBC nucleus mask ---" % (time.time() - start_time))

    im_nuc=imtools.overlayImage(im_resize,mask_nuc,(0,0,1),0.8,vis_diag=vis_diag,fig='nuc')    


    """
    CELL MASK
    """
    start_time = time.time() 
    
    mask_cell=detections.cell_mask(hsv_resize,diag.param,scale=scale,mask=mask_nuc,init_centers=diag.cent_init,vis_diag=vis_diag,fig='cell_mask')
    
    print("--- %s seconds - CELL mask ---" % (time.time() - start_time))

    """
    CELL MARKERS
    """
    start_time = time.time()  
    
    markers_rbc=detections.cell_markers_from_mask(mask_cell,diag.param,scale=scale,vis_diag=vis_diag,fig='cell_markers')         

    print("--- %s seconds - CELL markers ---" % (time.time() - start_time))

    im_rbc=imtools.overlayImage(im_resize,markers_rbc,(1,0,0),1,vis_diag=vis_diag,fig='sat')    


    """
    CREATE WBC REGIONS
    """
    label_nuc = measure.label(mask_nuc, connectivity=mask_nuc.ndim)
    
#    start_time = time.time()
   
    props = measure.regionprops(label_nuc)
#    print("--- %s seconds ---" % (time.time() - start_time))
 
    label_wbc=np.zeros(label_nuc.shape).astype('int32')
    props_large=props.copy()
    i_clean=0
    for ip, p in enumerate(props):
        if p.area<0.1*diag.param.rbcR**2*np.pi*scale**2:
            props_large.remove(p)
            continue
        if p.centroid[0]-p.major_axis_length<0 or p.centroid[0]+p.major_axis_length>im_resize.shape[0]:
            props_large.remove(p)
            continue
        if p.centroid[1]-p.major_axis_length<0 or p.centroid[1]+p.major_axis_length>im_resize.shape[1]:
            props_large.remove(p)
            continue
        i_clean+=1
        label_wbc[label_nuc==ip+1]=i_clean

    im_wbc=imtools.overlayImage(im_resize,label_wbc>0,(0,0,1),0.8,vis_diag=vis_diag,fig='sat')    
                       
    po=np.asarray([p.centroid for p in props_large])
    cent_dist=segmentations.center_diff_matrix(po,metric='euclidean')


    # TODO: handle 2 thresholds: distance and size

    # merging regions
    merge_pair_indices=np.argwhere(np.logical_and(cent_dist<2*diag.param.rbcR,cent_dist>0))
    
    for mi in merge_pair_indices:
        if mi[0]<mi[1] and\
            (props_large[mi[0]].area+props_large[mi[1]].area<2*diag.param.rbcR**2*np.pi*scale**2):
                label_wbc[label_wbc==mi[0]+1]=label_wbc[label_wbc==mi[1]+1].max()

    prop_wbc=measure.regionprops(label_wbc)
    diag.measures['n_WBC']=len(prop_wbc)
   
    """
    CREATE RBC REGIONS
    """    
    label_rbc = measure.label(markers_rbc, connectivity=mask_nuc.ndim)
    prop_rbc=measure.regionprops(label_rbc)
    diag.measures['n_RBC']=len(prop_rbc)
    
    
#    fig = plt.figure('detections')
#    ax = fig.add_subplot(111)
#    ax.imshow(im_resize)    
#    for p in prop_wbc:
#        cnt = draw.circle_perimeter(int(p.centroid[0]), int(p.centroid[1]), int(p.minor_axis_length))
#        cnt=(cnt[1][np.linspace(0,len(cnt[1])-1,21,dtype='uint8',endpoint=False)],cnt[0][np.linspace(0,len(cnt[0])-1,21,dtype='uint8',endpoint=False)]) # row,col -> x,y
#        ax.plot(cnt[0], cnt[1], '.', lw=1)
#   
#    diag.saveDiagFigure(fig,'wbc_detections',savedir=diag_dir)
#    plt.close(fig) 
    
    """
    CREATE shapes
    """
 
    shapelist=[]
    for p in prop_rbc:
         pts=[(p.centroid[1]/scale,p.centroid[0]/scale)]
         one_shape=('RBC','circle',pts,'None','None')
         shapelist.append(one_shape)
    for p in prop_wbc:
         pts=[(p.centroid[1]/scale+p.major_axis_length*np.cos(theta*2*np.pi/20)/scale,p.centroid[0]/scale+p.major_axis_length*np.sin(theta*2*np.pi/20)/scale) for theta in range(20)] 
         #pts=[(p.centroid[1]/scale,p.centroid[0]/scale)]
         one_shape=('WBC','circle',pts,'None','None')
         shapelist.append(one_shape)
    # Plot manual
    fig = plt.figure('detections',figsize=(20,20))
    fig=imtools.plotShapes(im,shapelist,color='r',\
                                   detect_shapes='ALL',text=('WBC'),fig=fig)

    diag.saveDiagFigure(fig,'detections',savedir=diag_dir)
    plt.close(fig) 