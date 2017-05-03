# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import os
import collections
import sys
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
import matplotlib.pyplot as plt
from matplotlib.path import Path
import argparse
import numpy as np

import annotations
import imtools
import cfg

def evaluate_wbc_detection(image_dir,output_dir,save_diag=False):
    
    plt.ioff()

    param=cfg.param()
    wbc_types=param.wbc_types
    wbc_basic_types=param.wbc_basic_types

    image_list_indir=imtools.imagelist_in_depth(image_dir,level=1)
    print('processing '+str(len(image_list_indir))+' images')

    detect_stat=[]
    for i, image_file in enumerate(image_list_indir):
        print(str(i)+' : '+image_file)
        """
        READ manual annotations
        """ 
        head, tail=os.path.splitext(image_file)
        xml_file_1=head+'.xml'
        if os.path.isfile(xml_file_1):
            try:
                xmlReader = annotations.AnnotationReader(xml_file_1)
                annotations_bb=xmlReader.getShapes()
            except:
                annotations_bb=[]
        else:
            annotations_bb=[]
        """
        READ auto annotations
        """            
        file_name=os.path.basename(image_file)
        head, tail=os.path.splitext(file_name)
        xml_file_2=os.path.join(output_dir,head+'.xml')
        if os.path.isfile(xml_file_2):
            try:
                xmlReader = annotations.AnnotationReader(xml_file_2)
                shapelist=xmlReader.getShapes()
            except:
                shapelist=[]
        else:
            shapelist=[]
          
        """
        READ image
        """
        im = io.imread(image_file) # read uint8 image
        
        """
        REMOVE ANNOTATIONS CLOSE TO BORDER
        """
        for each_bb in annotations_bb:
            bb=each_bb[2]
            if min((im.shape[1],im.shape[0])-np.average(bb,axis=0))<25 or min(np.average(bb,axis=0))<25:
                annotations_bb.remove(each_bb)
        """
        REMOVE ANNOTATIONS CLOSE TO BORDER
        """
        for each_bb in shapelist:
            bb=each_bb[2]
            if min((im.shape[1],im.shape[0])-np.average(bb,axis=0))<25 or min(np.average(bb,axis=0))<25:
                shapelist.remove(each_bb)
# TODO: add 25 as parameter
    
        if save_diag and (shapelist):
            fig = plt.figure('detections',figsize=(20,20),dpi=300)
            # Plot manual
            fig=imtools.plotShapes(im,annotations_bb,color='b',\
                                   detect_shapes=list(wbc_types.keys()),text='ALL',fig=fig)
            # Plot automatic
            fig=imtools.plotShapes(im,shapelist,color='r',\
                                   detect_shapes='ALL',text=list(wbc_basic_types.keys()),fig=fig)
            head, tail = str.split(os.path.abspath(xml_file_2),'.')
            detect_image_file=os.path.join(head+'_annotations.jpg')
            fig.savefig(detect_image_file,dpi=300)
            plt.close(fig)
        
        """
        COMPARE manual vs. automatic detections
        """
#        x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
#        x, y = x.flatten(), y.flatten()
#        points = np.vstack((x,y)).T
        
        if (shapelist) and (annotations_bb):       
            
            wbc_stat={}
            wbc_stat['wbc_annotated']=0
            for types in wbc_types.keys():
                wbc_stat['annotated_'+types]=0
            for types in wbc_basic_types.keys():
                wbc_stat['detected_'+types]=0
            wbc_stat['wbc_detected']=0
            wbc_stat['wbc_matched']=0
            wbc_stat['wbc_type_matched']=0
                    
                    
            for each_bb in annotations_bb:
                if each_bb[0] in list(wbc_types.keys()):
                    wbc_stat['wbc_annotated']+=1
                    for types in wbc_types.keys():
                        if each_bb[0]==types:
                            wbc_stat['annotated_'+types]+=1
                    
            for each_shape in shapelist:
                if each_shape[0] in list(wbc_basic_types.keys()):
                    wbc_stat['wbc_detected']+=1;
                    for types in list(wbc_basic_types.keys()):
                        if each_shape[0]==types:
                            wbc_stat['detected_'+types]+=1
                    for each_bb in annotations_bb:
                        if each_bb[0] in list(wbc_types.keys()):
                            bb=Path(each_bb[2])
                            center_point=[np.mean(each_shape[2],axis=0).tolist()]
                            intersect = bb.contains_points(center_point)    
                            if intersect.sum()>0:
                                p_over=intersect.sum()/len(center_point)
                                wbc_stat['wbc_matched']+=p_over
                                if each_bb[0]==each_shape[0]:
                                     wbc_stat['wbc_type_matched']+=p_over
                                annotations_bb.remove(each_bb)
                                break
                            
            detect_stat.append((image_file,wbc_stat))    

    """
    AGGREGATE STATISTICS
    """ 
        
    # initialize to zero
    wbc_total_stat={}
    wbc_total_stat['wbc_images']=len(image_list_indir)
    wbc_total_stat['wbc_images_ok']=len(detect_stat)
    if (detect_stat):
        for keys, values in detect_stat[0][1].items():
            wbc_total_stat[keys]=0
                       
    for stats in detect_stat:
        for keys, values in stats[1].items():
            wbc_total_stat[keys]+=values
                
    od = collections.OrderedDict(sorted(wbc_total_stat.items()))    
    for keys, values in od.items():
        print(keys+'\t: ',+values)
    if save_diag:
        with open(os.path.join(output_dir,'eval_stats_'+os.path.basename(image_dir)+'.txt'), 'wt',newline='') as f:
            w = csv.DictWriter(f, delimiter=':', fieldnames=['measures','values'])
            for keys, values in od.items():
                w.writerow({'measures' : keys, 'values' : values})

if __name__=='__main__':
    # Initialize argument parse object
    parser = argparse.ArgumentParser()

    # This would be an argument you could pass in from command line
    parser.add_argument('-m', action='store', dest='m', type=str, required=True,
                    default='')
    parser.add_argument('-a', action='store', dest='a', type=str, required=False,
                    default=None)
    parser.add_argument('-s', action='store', dest='s', type=bool, required=False,
                    default=False)
  

    # Parse the arguments
    inargs = parser.parse_args()
    path_str_m = os.path.abspath(inargs.m)
    path_str_a = os.path.abspath(inargs.a)
    
    evaluate_wbc_detection(image_dir=path_str_m,output_dir=path_str_a,save_diag=inargs.s)    
    sys.exit(1)