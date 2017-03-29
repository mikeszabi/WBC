# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""
import __init__
import os
import glob
import cfg

import classifications

import cell_classifier
import evaluate


#  %matplotlib qt5
 
##
param=cfg.param()
vis_diag=False

# data_dir=r'd:\DATA\Diagon_Test'

data_dir=None # access test data set

imDirs=os.listdir(param.getImageDirs(data_dir=data_dir))
print(imDirs)

# SELECT subdir
i_imDirs=0

diag_dir=param.getOutDir(dir_name='diag')
image_dir=param.getImageDirs(data_dir=data_dir,dir_name=imDirs[i_imDirs])
output_dir=os.path.join(param.getOutDir(dir_name='output'),imDirs[i_imDirs])


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)

# SELECT a TEST file
image_file=image_list_indir[15]

detect_stat=[]

cnn=classifications.cnn_classification()

for image_file in image_list_indir:
    # reading image
    
    print(image_file)
    
    """
    RUN automatic detection
    """
    shapelist=cell_classifier.cell_classifier(image_file,cnn=cnn,save_diag=True,out_dir=imDirs[i_imDirs])
    
"""
EVALUATION of DETECTION
"""

evaluate.evaluate_wbc_detection(image_dir,output_dir,save_diag=True)
    
