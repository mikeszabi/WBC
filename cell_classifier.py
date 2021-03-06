# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:21:59 2017

@author: SzMike
"""
import os
import sys
import argparse
import logging

import numpy as np;
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
# %matplotlib qt5
from matplotlib.path import Path

 
import imtools
import diagnostics
import detections
import annotations
import classifications 

# Logging setup
log_file='progress.log'
logging.basicConfig(filename=log_file,level=logging.DEBUG)

def batch_cell_classifier(image_dir,cnn=None,save_diag=False,out_dir=''): 
    
    
    if not os.path.exists(image_dir):
        logging.info('directory does not exists')
        return
    
    image_list_indir=imtools.imagelist_in_depth(image_dir,level=1)
    logging.info('processing '+str(len(image_list_indir))+' images')
        
    for image_file in image_list_indir:    
        logging.info(image_file)
        cell_classifier(image_file,cnn=cnn,save_diag=save_diag,out_dir=out_dir)
    
    logging.info('DONE')
    
def cell_classifier(image_file,cnn=None,save_diag=False,out_dir=''): 
    
    vis_diag=False
    
# OPEN THE image to be processed
    try:
        im = io.imread(image_file) # read uint8 image
    except Exception:
        logging.info(image_file+' does not exist')
        return []
    if im.ndim!=3:
        logging.info('not color image')
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
    sat_tsh=max(diag.sat_q95,diag.param.wbc_min_sat)
    mask_nuc=detections.wbc_nucleus_mask(hsv_resize,diag.param,sat_tsh=sat_tsh,scale=scale,vis_diag=vis_diag,fig='')
    """
    CREATE WBC REGIONS
    """    
    prop_wbc=detections.wbc_regions(mask_nuc,diag.param,scale=scale)
  
    """
    CELL FOREGORUND MASK
    """    
    mask_cell=detections.cell_mask(hsv_resize,diag.param,scale=scale,mask=mask_nuc,init_centers=diag.cent_init,vis_diag=vis_diag,fig='cell_mask')
    
    """
    CELL MARKERS AnD REGIONS
    """    
    markers_rbc, prop_rbc=detections.cell_markers_from_mask(mask_cell,diag.param,scale=scale,vis_diag=vis_diag,fig='cell_markers')         
   
    """
    COUNTING
    """
    diag.measures['n_WBC']=len(prop_wbc)
    diag.measures['n_RBC']=len(prop_rbc)
    
    """
    PARAMETERS for WBC NORMALIZATION 
    """
    if mask_nuc.sum()>0:
        pixs=im_resize[mask_nuc,]
        diag.measures['nucleus_median_rgb']=np.median(pixs,axis=0)
       
    """
    CHECK ERRORS
    """
    diag.checks()
    if len(diag.error_list)>0:
        logging.info(image_file+' is of wrong quality')
        diag.writeDiagnostics(diag_dir)
        return []
    
    """
    CREATE shapes
    """
    shapelist_WBC=[]
    for p in prop_wbc:
        # centroid is in row,col
         pts=[(p.centroid[1]/scale+0.8*p.major_axis_length*np.cos(theta*2*np.pi/20)/scale,p.centroid[0]/scale+0.8*p.major_axis_length*np.sin(theta*2*np.pi/20)/scale) for theta in range(20)] 
         #pts=[(p.centroid[1]/scale,p.centroid[0]/scale)]
         one_shape=('None','circle',pts,'None','None')
         
         # WBC classification
         # check if shape is fully contained in the image canvas
#         if min((im.shape[1],im.shape[0])-np.max(one_shape[2],axis=0))<0\
#                or min(np.min(one_shape[2],axis=0))<0:
#            continue

         
         im_cropped,o,r=imtools.crop_shape(diag.im_corrected,one_shape,\
                                            diag.param.rgb_norm,diag.measures['nucleus_median_rgb'],\
                                            scale=1,adjust=True)
         if im_cropped is not None and cnn is not None:
             # do the actual classification
             if r[0] > 1.5*diag.param.rbcR:
                 wbc_label, pct=cnn.classify(im_cropped)
             else:
                 wbc_label=['un']
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
    
    logging.info('STARTING')
    
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

# create classfier
    cnn=classifications.cnn_classification()

# Parse the arguments
    inargs = parser.parse_args()
    path_str = os.path.abspath(inargs.i)
   
    if inargs.b is None:
        logging.info('Single image process')
        cell_classifier(path_str,cnn=cnn,save_diag=inargs.s,out_dir=inargs.o)
    else:
        logging.info('Batch execution')
        batch_cell_classifier(path_str,cnn=cnn,save_diag=inargs.s,out_dir=inargs.o)    
    
    # deleting log file
    logging.info('FINISHING')
    logging.shutdown()
    os.remove(log_file)
    sys.exit(1)
  