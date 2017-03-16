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

import __init__
import csv
from collections import Counter
import random
import math
import os
import numpy as np

import cfg

param=cfg.param()

data_dir=r'C:\Users\SzMike\OneDrive\WBC\DATA'
image_dir=os.path.join(data_dir,'Detected_Cropped')
train_dir=os.path.join(data_dir,'Training')
train_image_list_file=os.path.join(train_dir,'images_train.csv')
test_image_list_file=os.path.join(train_dir,'images_test.csv')
image_data=os.path.join(image_dir,'detections.csv')


trainRatio=0.7


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

reader =csv.DictReader(open(image_data, 'rt'), delimiter=';')
samples = {}

for row in reader:
    wbc_type='0'
    for bt in param.wbc_basic_types:
        if bt in row['wbc']:
            wbc_type=param.wbc_basic_types[bt]
            break
    samples[row['crop']]=wbc_type


sampleCount=Counter(samples.values())
# remove prods with less than 10 occurencies
i=0
testProds = {}
trainProds = {}

# TODO: enrichment?
min_count=200
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

