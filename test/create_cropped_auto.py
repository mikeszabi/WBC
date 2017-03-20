# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:59:53 2017

@author: SzMike
"""

import __init__
import os
import skimage.io as io
from skimage import measure
from skimage import exposure
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from csv import DictWriter


import annotations
import diagnostics
import imtools
import cfg
import detections

user='SzMike'
#user='mikeszabi'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WBC\DATA')
image_dir=os.path.join(output_base_dir,'Annotated')
output_dir=os.path.join(output_base_dir,'Detected_Cropped')
#mask_dir=os.path.join(output_base_dir,'Mask')

image_dir=r'd:\Projects\WBC\data\Test'
output_dir=r'd:\Projects\WBC\diag'


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
    WBC nucleus masks
    """    
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=diag.sat_q95,scale=scale,vis_diag=False,fig='')
   
    """
    CREATE WBC REGIONS
    """    
    prop_wbc=detections.wbc_regions(mask_nuc,diag.param,scale=scale)
    
    """
    SAVE NUCLEUS MASK
    """
#    mask_file_name=str.replace(image_file,image_dir,mask_dir)
#    if not os.path.isdir(os.path.dirname(mask_file_name)):
#        os.makedirs(os.path.dirname(mask_file_name))
#    io.imsave(mask_file_name,255*mask_nuc)

    """
    PARAMETERS for WBC NORMALIZATION 
    """
    pixs=im_resize[mask_nuc,]
    diag.measures['nucleus_median_rgb']=np.median(pixs,axis=0)

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
    CREATE SHAPES    
    """

    shapelist_WBC=[]
    for p in prop_wbc:
        # centroid is in row,col
         pts=[(p.centroid[1]/scale+0.75*p.major_axis_length*np.cos(theta*2*np.pi/20)/scale,p.centroid[0]/scale+0.75*p.major_axis_length*np.sin(theta*2*np.pi/20)/scale) for theta in range(20)] 
         #pts=[(p.centroid[1]/scale,p.centroid[0]/scale)]
         one_shape=('WBC','circle',pts,'None','None')
         shapelist_WBC.append(one_shape)    
     
    for shapes in shapelist_WBC:
        is_pos_detect=False
        i_detected+=1
        # centroid is in row,col
        mins=(np.min(shapes[2],axis=0)*scale).astype('int32')
        maxs=(np.max(shapes[2],axis=0)*scale).astype('int32')
        o=(mins+maxs)/2
        r=(maxs-mins)/2
        if min(mins)>=0 and maxs[1]<im_resize.shape[0] and maxs[0]<im_resize.shape[0]:
            im_cropped=im_resize[max(mins[1],0):min(maxs[1],im_resize.shape[0]-1),\
                                max(mins[0],0):min(maxs[0],im_resize.shape[1]-1)]
            mask_cropped=mask_nuc[max(mins[1],0):min(maxs[1],im_resize.shape[0]-1),\
                                max(mins[0],0):min(maxs[0],im_resize.shape[1]-1)]
            wbc_type='fp'
            for each_bb in annotations_bb:
    #            if each_bb[0] in list(wbc_types.keys()):
                    # only if in wbc list to be detected
                bb=Path(scale*np.array(each_bb[2]))
                intersect = bb.contains_point(np.asarray(o)) 
                if intersect:
                    # automatic detection is within manual annotation
                    is_pos_detect=True   
                    wbc_type=each_bb[0]
                    if wbc_type not in list(wbc_types.keys()):
                        wbc_type='fp' # false positive
                        break
            
            # Normalization
            adjusted_cropped=im_cropped.copy()
            # local to cropped
#            pixs=im_cropped[mask_cropped>0,]
#            nuc_med_rgb=np.median(pixs,axis=0)
            # global to image
            nuc_med_rgb=diag.measures['nucleus_median_rgb']
            
            gamma=np.zeros(3)
            gain=np.zeros(3)
            for ch in range(3):
                gamma[ch]=np.log(255-diag.param.rgb_norm[ch])/np.log(255-nuc_med_rgb[ch])
                gain[ch]=diag.param.rgb_norm[ch]/np.power(nuc_med_rgb[ch],gamma[ch])
            adjusted_cropped=exposure.adjust_gamma(adjusted_cropped,np.mean(gamma),np.mean(gain))

                
        
            
            # SAVE    
            crop_file=os.path.join(output_dir,wbc_type+'_'+str(i_detected)+'.png')
            io.imsave(crop_file,im_cropped)
            
            sample={'im':os.path.basename(image_file),'crop':os.path.basename(crop_file),\
                    'rbcR':diag.param.rbcR,'wbc':wbc_type,\
                    'scale':scale,'origo':np.asarray(o)/scale,'radius':r/scale,\
                    'sat_tsh':diag.measures['saturation_q95'],\
                    'nucleus_median_rgb':diag.measures['nucleus_median_rgb']}
            samples.append(sample)
        
keys = samples[0].keys()
with open(os.path.join(output_dir,'detections.csv'), "w", newline='') as f:
    dict_writer = DictWriter(f, keys, delimiter=";")
    dict_writer.writeheader()
    for sample in samples:
        dict_writer.writerow(sample)