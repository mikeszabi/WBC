# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""
import _init_path
import os
import skimage.io as io
import glob
import cfg
from matplotlib.path import Path
import numpy as np

import annotations
import cell_detector
import imtools

#%matplotlib qt5
 
##
param=cfg.param()
vis_diag=False

imDirs=os.listdir(param.getImageDirs(''))
print(imDirs)
i_imDirs=-1
output_dir=param.getOutDir('output')
diag_dir=param.getOutDir('diag')

image_dir=param.getImageDirs(imDirs[1])

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

print(image_list_indir)

detect_stat=[]

for image_file in image_list_indir:
    # reading image
    
    # image_file=image_list_indir[0]
    # image_file=image_dir+'\\36.bmp'
    print(image_file)
    
    """
    RUN automatic detection
    """
    shapelist=cell_detector.main(image_file)
    
    """
    READ manual annotations
    """
    head, tail=os.path.splitext(image_file)
    xml_file=head+'.xml'
    if os.path.isfile(xml_file):
        xmlReader = annotations.AnnotationReader(xml_file)
        annotations_bb=xmlReader.getShapes()
        n_wbc=len(annotations_bb)
    else:
        annotations_bb=[]
        n_wbc=-1

     
    im = io.imread(image_file) # read uint8 image
   
    if vis_diag:
        f=imtools.plotShapes(im,annotations_bb,color='b',text=False)
        f=imtools.plotShapes(im,shapelist,fig=f)
        
    """
    COMPARE manual vs. automatic detections
    """
    x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    
    n_wbc_detected=0
    n_wbc_matched=0
    for each_shape in shapelist:
        if each_shape[0]=='WBC':
            n_wbc_detected+=1;
            for each_bb in annotations_bb:
                bb=Path(each_bb[2])
                intersect = bb.contains_points(each_shape[2])    
                if intersect.sum()>0:
                    p_over=intersect.sum()/len(each_shape[2])
                    n_wbc_matched+=p_over
                    annotations_bb.remove(each_bb)
                    break
                    
    detect_stat.append((image_file,n_wbc,n_wbc_detected,n_wbc_matched))    