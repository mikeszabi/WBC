# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:59:53 2017

@author: SzMike
"""

import __init__
import os
import skimage.io as io
from skimage import measure
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from csv import DictWriter


import annotations
import diagnostics
import imtools
import cfg
import detections


output_base_dir=r'C:\Users\mikeszabi\OneDrive\WBC\DATA'
image_dir=os.path.join(output_base_dir,'Annotated')
output_dir=os.path.join(output_base_dir,'Detected_Cropped')
mask_dir=os.path.join(output_base_dir,'Mask')

plt.ioff()

param=cfg.param()
wbc_types=param.wbc_types

image_list_indir=imtools.imagelist_in_depth(image_dir,level=1)
print('processing '+str(len(image_list_indir))+' images')

i_detected=-1
samples=[]
for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)
    """
    READ IMAGE
    """
    # READ THE IMAGE
    im = io.imread(image_file) # read uint8 image
   
    """ 
    CREATE AND SAVE DIAGNOSTICS
    """
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
    
    """
    CREATE HSV
    """
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,diag.param.middle_size,interpolation = 0)
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,diag.param.middle_size,interpolation = 0)

    """
    DETECT AND LABEL WBC NUCLEUS
    """
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=diag.sat_q95,scale=scale,vis_diag=False,fig='')
    label_nuc = measure.label(mask_nuc, connectivity=mask_nuc.ndim)

    mask_file_name=str.replace(image_file,image_dir,mask_dir)
    if not os.path.isdir(os.path.dirname(mask_file_name)):
        os.makedirs(os.path.dirname(mask_file_name))
    io.imsave(mask_file_name,255*mask_nuc)

    """
    CALCULATE NUCLEUS SATURATION
    """
    sat=hsv_resize[:,:,1]
    sat=sat[mask_nuc]
    val=hsv_resize[:,:,2]
    val=val[mask_nuc]
    diag.measures['sat_nucleus_median']=np.median(sat)
    diag.measures['sat_nucleus_std']=np.std(sat)
    diag.measures['val_nucleus_median']=np.median(val)
    """
    READ manual annotations
    """ 
    head, tail=os.path.splitext(image_file)
    xml_file_1=head+'.xml'
    if os.path.isfile(xml_file_1):
        try:
            xmlReader = annotations.AnnotationReader(xml_file_1)
            annotations_bb=xmlReader.getShapes()
        except:
            annotations_bb=[]
            break
    else:
        annotations_bb=[]
        break    

    """
    CHECK WBC DETECTIONS
    """
    
    props = measure.regionprops(label_nuc)
    props_large=props.copy()
    
    for p in props:
        if p.area<0.33*diag.param.rbcR**2*np.pi*scale**2:
            props_large.remove(p)
            continue
        if p.centroid[0]-p.major_axis_length<0 or p.centroid[0]+p.major_axis_length>im_resize.shape[0]:
            props_large.remove(p)
            continue
        if p.centroid[1]-p.major_axis_length<0 or p.centroid[1]+p.major_axis_length>im_resize.shape[1]:
            props_large.remove(p)
            continue
    
    for p in props_large:
        is_pos_detect=False
        i_detected+=1
        o=(int(p.centroid[1]),int(p.centroid[0])) # centroid in row/col, origo in x/y
        r=int(p.major_axis_length)
        im_cropped=im_resize[o[1]-r:o[1]+r,o[0]-r:o[0]+r]
        wbc_type='fp'
        for each_bb in annotations_bb:
#            if each_bb[0] in list(wbc_types.keys()):
                # only if in wbc list to be detected
            bb=Path(scale*np.array(each_bb[2]))
            intersect = bb.contains_point(np.asarray(o)) 
            if intersect:
                # automatic detection is within automatic annotation
                is_pos_detect=True   
                wbc_type=each_bb[0]
                if wbc_type not in list(wbc_types.keys()):
                    wbc_type='fp' # false positive
                    break
        crop_file=os.path.join(output_dir,wbc_type+'_'+str(i_detected)+'.png')
        
        io.imsave(crop_file,im_cropped)
        
        sample={'im':os.path.basename(image_file),'crop':os.path.basename(crop_file),\
                'rbcR':diag.param.rbcR,'wbc':wbc_type,\
                'scale':scale,'origo':np.asarray(o)/scale,'radius':r/scale,\
                'sat_tsh':diag.measures['saturation_q95'],\
                'sat_nucleus_median':diag.measures['sat_nucleus_median'],\
                'val_nucleus_median':diag.measures['val_nucleus_median']}
        samples.append(sample)
        
keys = samples[0].keys()
with open(os.path.join(output_dir,'detections.csv'), "w", newline='') as f:
    dict_writer = DictWriter(f, keys, delimiter=";")
    dict_writer.writeheader()
    for sample in samples:
        dict_writer.writerow(sample)