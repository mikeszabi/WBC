# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:29:36 2016

@author: SzMike
"""

# https://github.com/Microsoft/CNTK/blob/v2.0.beta6.0/Tutorials/CNTK_201A_CIFAR-10_DataLoader.ipynb


from PIL import Image
import numpy as np
import os
import xml.etree.cElementTree as et
import xml.dom.minidom
import csv

#%matplotlib inline

# Config matplotlib for inline plotting

imgSize = 32
numFeature = imgSize * imgSize * 3

# Paths for saving the text files

data_dir=r'C:\Users\SzMike\OneDrive\WBC\DATA'
image_dir=os.path.join(data_dir,'Detected_Cropped')

train_dir=os.path.join(data_dir,'Training')
train_image_list_file=os.path.join(train_dir,'images_train.csv')
test_image_list_file=os.path.join(train_dir,'images_test.csv')

train_filename = os.path.join(train_dir,'Train_cntk_text.txt')
test_filename = os.path.join(train_dir,'Test_cntk_text.txt')

train_img_directory = os.path.join(train_dir,'/Train')
test_img_directory = os.path.join(train_dir,'/Test')

train_map=os.path.join(train_dir,'train_map.txt')
test_map=os.path.join(train_dir,'test_map.txt')
train_regr_labels=os.path.join(train_dir,'train_regrLabels.txt')

data_mean_file=os.path.join(train_dir,'data_mean.xml')

def saveImage(fname, pixData, label, mapFile, regrFile, pad, **key_parms):

    if ('mean' in key_parms):
        key_parms['mean'] += pixData

    if pad > 0:
        pixData = np.pad(pixData, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=128) 

    img = Image.new('RGB', (imgSize + 2 * pad, imgSize + 2 * pad))
    pixels = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixels[x, y] = (pixData[0][y][x], pixData[1][y][x], pixData[2][y][x])
    img.save(fname)
    mapFile.write("%s\t%d\n" % (fname, label))
    
    # compute per channel mean and store for regression example
    channelMean = np.mean(pixData, axis=(1,2))
    regrFile.write("|regrLabels\t%f\t%f\t%f\n" % (channelMean[0]/255.0, channelMean[1]/255.0, channelMean[2]/255.0))
    
def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgSize)
    et.SubElement(root, 'Col').text = str(imgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imgSize * imgSize * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))

def saveTrainImages(filename, train_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    data = {}
    dataMean = np.zeros((3, imgSize, imgSize)) # mean is in CHW format.
    
# Szabi code
    reader =csv.DictReader(open(filename, 'rt'), delimiter=';')
    prods = {}

    with open(os.path.join(train_dir,'train_map.txt'), 'w') as mapFile:
        with open(os.path.join(train_dir,'train_regrLabels.txt'), 'w') as regrFile:
            for row in reader:
                try:
                    prods[row['image']]=row['category']
# read image file
# create data sequence RRR GGG BBB

# TODO: check THIS PART!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    fname = str.replace(row['image'],'mikeszabi','SzMike')
                    im = Image.open(fname)
                    im.thumbnail([imgSize, imgSize], Image.ANTIALIAS)
                    fname = os.path.join(train_dir,os.path.basename(row['image'])) # .decode('utf-8')  -solves unicode problem
                    data=np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
                    data=np.transpose(data, (2, 0, 1)) # CHW format.
                    saveImage(fname, data, int(row['category']), mapFile, regrFile, 4, mean=dataMean)
                except:
                    print(row)
    dataMean = dataMean / len(prods)
    saveMean(os.path.join(train_dir,'data_mean.xml'), dataMean)

def saveTestImages(filename, train_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    # Szabi code
    reader =csv.DictReader(open(filename, 'rt'), delimiter=';')
    prods = {}

    with open(os.path.join(train_dir,'test_map.txt'), 'w') as mapFile:
        with open(os.path.join(train_dir,'test_regrLabels.txt'), 'w') as regrFile:
            for row in reader:
                try:
                    prods[row['image']]=row['category']
# read image file
# create data sequence RRR GGG BBB

# TODO: check THIS PART!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    fname = str.replace(row['image'],'mikeszabi','SzMike')
                    im = Image.open(fname)
                    im.thumbnail([imgSize, imgSize], Image.ANTIALIAS)
                    fname = os.path.join(train_dir,os.path.basename(row['image'])) # .decode('utf-8')  -solves unicode problem
                    data=np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
                    data=np.transpose(data, (2, 0, 1)) # CHW format.
                    saveImage(fname, data, int(row['category']), mapFile, regrFile, 0)
                except:
                    print(row)


print ('Converting train data to png images...')
saveTrainImages(train_image_list_file, train_dir)
print ('Done.')
print ('Converting test data to png images...')
saveTestImages(test_image_list_file, train_dir)
print ('Done.')
