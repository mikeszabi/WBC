# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:02:17 2017

@author: SzMike
"""

import __init__
import os
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
import glob
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from csv import DictWriter


import annotations
import diagnostics
import imtools


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']

data_dir=r'd:\DATA\DiagonAdatbazis_20170221-5'
subdirs=glob.glob(os.path.join(data_dir,'*'))

samples=[]

for sd in subdirs:
    if os.path.isdir(sd):
        image_list_indir = []
        for ext in included_extenstions:
            image_list_indir.extend(glob.glob(os.path.join(sd, ext)))

            for image_file in image_list_indir:

                print(image_file)    
                # READ THE IMAGE
                im = io.imread(image_file) # read uint8 image
   
                diag=diagnostics.diagnostics(im,image_file,vis_diag=False)
                output_dir=diag.param.getOutDir(dir_name='train')

                head, tail=os.path.splitext(image_file)
                xml_file_1=head+'.xml'
                if os.path.isfile(xml_file_1):
                    try:
                        xmlReader = annotations.AnnotationReader(xml_file_1)
                        annotations_bb=xmlReader.getShapes()
                        n_wbc=len(annotations_bb)
                    except:
                        continue
                else:
                    continue
                
                fname=os.path.basename(image_file)
                fid=fname=os.path.split(sd)[-1]+'__'+os.path.basename(image_file)
                
                for i, each_bb in enumerate(annotations_bb):
                    minx=min(c[0] for c in each_bb[2])
                    miny=min(c[1] for c in each_bb[2])
                    maxx=max(c[0] for c in each_bb[2])
                    maxy=max(c[1] for c in each_bb[2])
                    
                    if sum(c[0]==minx for c in each_bb[2])!=2 or sum(c[1]==miny for c in each_bb[2])!=2 or\
                         sum(c[0]==maxx for c in each_bb[2])!=2 or sum(c[1]==maxy for c in each_bb[2])!=2:                       
                         sample={'im':image_file,'r':diag.param.rbcR,'wbc':'wrong',\
                                'minx':minx,'maxx':maxx,'dx':maxx-minx,\
                                'miny':miny,'maxy':maxy,'dy':maxy-miny}
                         samples.append(sample)
                         continue
                    else:
                        sample={'im':image_file,'r':diag.param.rbcR,'wbc':each_bb[0],\
                                'minx':minx,'maxx':maxx,'dx':maxx-minx,\
                                'miny':miny,'maxy':maxy,'dy':maxy-miny}
                        samples.append(sample)
                        im_cropped=im[miny:maxy,minx:maxx,:]
                        io.imsave(os.path.join(output_dir,each_bb[0]+'_'+str(i)+'_'+fid),im_cropped)
                    
#            fig = plt.figure()
#            fig=imtools.plotShapes(im,annotations_bb,color='b',text='ALL',fig=fig)
#    

keys = samples[0].keys()
with open(os.path.join(output_dir,'bbs.csv'), "w", newline='') as f:
    dict_writer = DictWriter(f, keys, delimiter=";")
    dict_writer.writeheader()
    for sample in samples:
        dict_writer.writerow(sample)