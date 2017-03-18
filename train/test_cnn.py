# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:55:49 2017

@author: SzMike
"""


import __init__
import pandas as pd
import os
import numpy as np
import skimage.io as io
from PIL import Image


import cfg
import imtools

#test_minibatch_size = 1000
#
#sample_count = 0
#test_results = []
#num_test_samples=100
#
#while sample_count < num_test_samples:
#
#    minibatch = test_minibatch_source.next_minibatch(min(test_minibatch_size, num_test_samples - sample_count))
#
#    # Specify the mapping of input variables in the model to actual minibatch data to be tested with
#    data = {input_vars: minibatch[test_features],
#            labels: minibatch[test_labels]}
#    eval_error = trainer.test_minibatch(data)
#    test_results.append(eval_error)
#
#    sample_count += data[labels].num_samples
#
## Printing the average of evaluation errors of all test minibatches
#print("Average errors of all test minibatches: %.3f%%" % (float(np.mean(test_results, dtype=float))*100))

imgSize=32
num_classes  = 6

param=cfg.param()

data_dir=r'C:\Users\SzMike\OneDrive\WBC\DATA'
image_dir=os.path.join(data_dir,'Detected_Cropped')

#image_data=os.path.join(image_dir,'detections.csv')
image_data=os.path.join(data_dir,'Training','images_test.csv')

image_mean   = 128


def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

df = pd.read_csv(image_data,delimiter=';')
samples = {}
contingency_table=np.zeros((num_classes,num_classes))
for i, im_name in enumerate(df['image']):
#    i=200
    image_file=os.path.join(image_dir,im_name)    
#    image_file=r'C:\Users\SzMike\OneDrive\WBC\DATA\Training\Train\ne_50.png'

#    wbc_type='0'
#    for bt in param.wbc_basic_types:
#        if bt in df['category'][i]:
    label=df['category'][i]
    wbc_type=keysWithValue(param.wbc_basic_types,str(df['category'][i]))
    if wbc_type==[]:
        wbc_type='0'
    im=io.imread(image_file)
    data,scale=imtools.imRescaleMaxDim(im,imgSize, boUpscale = True, interpolation = 0)
    rgb_image=data.astype('float32')
    rgb_image  -= image_mean
    bgr_image = rgb_image[..., [2, 1, 0]]
    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
    
#    rgb_image = np.asarray(Image.open(image_file), dtype=np.float32) - 128
#    bgr_image = rgb_image[..., [2, 1, 0]]
#    pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
       
    result  = np.round(np.squeeze(pred.eval({pred.arguments[0]:[pic]}))*100)
    predicted_label=np.argmax(result)
    contingency_table[predicted_label,label]+=1
#    print(df['wbc'][i])
#    print(result)
#    print(keysWithValue(param.wbc_basic_types,str(mr)))
#    plt.imshow(im)
#    
