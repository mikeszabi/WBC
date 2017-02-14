# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""
import _init_path
import os
import glob
import cfg
import cell_detector

#%matplotlib qt5
 
##
param=cfg.param()

imDirs=os.listdir(param.getTestImageDirs(''))
print(imDirs)
i_imDirs=-1
output_dir=param.getOutDir('output')
diag_dir=param.getOutDir('diag')

image_dir=param.getTestImageDirs(imDirs[i_imDirs])

included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

print(image_list_indir)

for image_file in image_list_indir:
    # reading image
    
    #image_file=image_list_indir[1]
    #image_file=image_dir+'\\36.bmp'
    print(image_file)
    
    cell_detector.main(image_file)