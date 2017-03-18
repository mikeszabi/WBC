# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:21:59 2017

@author: SzMike
"""
import os
import sys
import argparse

import numpy as np;
import skimage.io as io
from skimage import measure
# %matplotlib qt5
 
import __init__
import imtools
import diagnostics
import detections
import annotations


def batch_cell_detector(image_dir,save_diag=False,out_dir=''): 
    
    if not os.path.exists(image_dir):
        print('directory does not exists')
        return
    
    image_list_indir=imtools.imagelist_in_depth(image_dir,level=1)
    print('processing '+str(len(image_list_indir))+' images')
        
    for image_file in image_list_indir:    
        print(image_file)
        cell_detector(image_file,save_diag,out_dir=out_dir)

def cell_detector(image_file,save_diag=False,out_dir=''): 
    
    vis_diag=False
    
# OPEN THE image to be processed
    try:
        im = io.imread(image_file) # read uint8 image
    except Exception:
        print(image_file+' does not exist')
        return []
    if im.ndim!=3:
        print('not color image')
        return []    
   
    """
    DIAGNOSTICS
    """                     
    diag=diagnostics.diagnostics(im,image_file,vis_diag=False)        
    """
    CREATE OUTPUT AND DIAG DIRS
    """
       
    output_dir=diag.param.getOutDir(dir_name=os.path.join('output',out_dir))
    diag_dir=diag.param.getOutDir(dir_name=os.path.join('diag',out_dir))
    
    """
    RESIZE
    """    
    hsv_resize, scale=imtools.imRescaleMaxDim(diag.hsv_corrected,diag.param.middle_size,interpolation = 0)
    im_resize, scale=imtools.imRescaleMaxDim(diag.im_corrected,diag.param.middle_size,interpolation = 0)
    """
    WBC nucleus masks
    """    
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=diag.sat_q95,scale=scale,vis_diag=False,fig='')
    """
    CELL FOREGORUND MASK
    """    
    mask_cell=detections.cell_mask(hsv_resize,diag.param,scale=scale,mask=mask_nuc,init_centers=diag.cent_init,vis_diag=vis_diag,fig='cell_mask')
    
    """
    CELL MARKERS AnD REGIONS
    """    
    markers_rbc, prop_rbc=detections.cell_markers_from_mask(mask_cell,diag.param,scale=scale,vis_diag=vis_diag,fig='cell_markers')         
    """
    CREATE WBC REGIONS
    """    
    prop_wbc=detections.wbc_regions(mask_nuc,diag.param,scale=scale)
      
    """
    CREATE RBC REGIONS
    """    
    
    diag.measures['n_WBC']=len(prop_wbc)
    diag.measures['n_RBC']=len(prop_rbc)
    
   
    """
    CHECK ERRORS
    """
    diag.checks()
    if len(diag.error_list)>0:
        print(image_file+' is of wrong quality')
        return []
    
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
    
    head, tail=os.path.split(image_file)
    xml_file=os.path.join(output_dir,tail.replace('.bmp',''))
    tmp = annotations.AnnotationWriter(head,xml_file, (im.shape[0],im.shape[1]))
    tmp.addShapes(shapelist)
    tmp.save()

    """
    CREATE and SAVE DIAGNOSTICS IMAGES
    """
    if save_diag:
        im_rbc=imtools.overlayImage(im_resize,markers_rbc,(1,0,0),1,vis_diag=vis_diag,fig='sat')    
        im_detect=imtools.overlayImage(im_rbc,mask_nuc,(0,0,1),0.5,vis_diag=vis_diag,fig='nuc')    
    
#        border=np.zeros(im_resize.shape[0:2]).astype('uint8')
#        border[0:diag.param.middle_border,:]=1
#        border[-diag.param.middle_border-1:-1,:]=1     
#        border[:,0:diag.param.middle_border]=1
#        border[:,-diag.param.middle_border-1:-1]=1    
#        im_detect=imtools.overlayImage(im_nuc,border>0,\
#                (1,1,0),0.2,vis_diag=vis_diag,fig='detections')       
#        im_detect,scale=imtools.imRescaleMaxDim(im_detect,diag.param.middle_size,interpolation = 1)
#        
        diag.saveDiagImage(im_detect,'detections',savedir=diag_dir)
        
        diag.writeDiagnostics(diag_dir)   


    return shapelist

if __name__=='__main__':
# Initialize argument parse object
    parser = argparse.ArgumentParser()

    # This would be an argument you could pass in from command line
    parser.add_argument('-i', action='store', dest='i', type=str, required=True,
                    default='')
    parser.add_argument('-b', action='store', dest='b', type=str, required=False,
                    default=None)
    parser.add_argument('-s', action='store', dest='s', type=bool, required=False,
                    default=False)
    parser.add_argument('-o', action='store', dest='o', type=str, required=False,
                    default='')

# Parse the arguments
    inargs = parser.parse_args()
    path_str = os.path.abspath(inargs.i)
    
    if inargs.b is None:
        print('Single image process')
        cell_detector(path_str,save_diag=inargs.s==inargs.s,out_dir=inargs.o)
    else:
        print('Batch execution')
        batch_cell_detector(path_str,save_diag=inargs.s==inargs.s,out_dir=inargs.o)    
    sys.exit(1)
  