# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:40:14 2016

@author: picturio
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:18:47 2016

@author: SzMike
"""
import glob
import csv
from collections import Counter
import random
import math
import os

import cfg


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

# SETTING PARAMETERS and DIRS

param=cfg.param()
trainRatio=0.75


included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']


user='picturio'
output_base_dir=os.path.join(r'C:\Users',user,'OneDrive\WBC\DATA')

image_dirs=[os.path.join(output_base_dir,'Detected_Cropped'),\
            os.path.join(output_base_dir,'Detected_Cropped_ba')]

train_dir=os.path.join(output_base_dir,'Training')
train_image_list_file=os.path.join(train_dir,'images_train.csv')
test_image_list_file=os.path.join(train_dir,'images_test.csv')

# COUNTING TYPES
#samples = {}
#for image_dir in image_dirs:
#    image_data=os.path.join(image_dir,'detections.csv')
#    reader =csv.DictReader(open(image_data, 'rt'), delimiter=';')
#    for row in reader:
#        wbc_type='0'
#        for bt in param.wbc_basic_types:
#            if bt in row['wbc']:
#                wbc_type=param.wbc_basic_types[bt]
#                break
#        samples[os.path.join(image_dir,row['crop'])]=wbc_type

samples = {}

for image_dir in image_dirs:
    image_list_indir = []
    for ext in included_extenstions:
        image_list_indir.extend(glob.glob(os.path.join(image_dir, ext)))

    for i, image_file in enumerate(image_list_indir):
        file_name=os.path.basename(image_file)
        #print(str(i)+' : '+os.path.basename(file_name))
        wbc_type=file_name.split('_')[0]
        for bt in param.wbc_basic_types:
            if bt in wbc_type:
                samples[image_file]=param.wbc_basic_types[bt]

sampleCount=Counter(samples.values())

# CREATE TEST AND TRAIN LIST USING RANDOM SPLIT
i=0
testProds = {}
trainProds = {}

for cat, count in sampleCount.items():
    catProds=keysWithValue(samples,cat)
    random.shuffle(catProds)
    splitInd=int(math.ceil(trainRatio*len(catProds)))
    trainItems=catProds[:splitInd]
    testItems=catProds[splitInd:]
    for item in testItems:
        testProds[item]=cat
    for item in trainItems:
        trainProds[item]=cat

out = open(train_image_list_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['image','category'])
w.writeheader()
for key, value in trainProds.items():
    w.writerow({'image' : key, 'category' : value})
out.close()

out = open(test_image_list_file, 'wt')
w = csv.DictWriter(out, delimiter=';', fieldnames=['image','category'])
w.writeheader()
for key, value in testProds.items():
    w.writerow({'image' : key, 'category' : value})
out.close()

