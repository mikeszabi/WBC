# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:13:20 2017

@author: SzMike
"""
import os
import glob

import evaluate


#  %matplotlib qt5
 
##



image_dir=r'd:\DATA\Diagon\Test_3\manual' # manual
output_dir=r'd:\DATA\Diagon\Test_3\automatic' # automatic


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
image_list_indir = []
for ext in included_extenstions:
   image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

for i, image_file in enumerate(image_list_indir):
    print(str(i)+' : '+image_file)

# SELECT a TEST file
image_file=image_list_indir[13]



"""
EVALUATION of DETECTION
"""

evaluate.evaluate_wbc_detection(image_dir,output_dir,save_diag=True)
    
