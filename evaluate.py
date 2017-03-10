# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import __init__
import os
import skimage.io as io
import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
import argparse
import numpy as np

import annotations
import imtools

def evaluate_wbc_detection(image_dir,output_dir,save_diag=False):
    
    plt.ioff()

    wbc_types={'bne':'Band neutrophiles',\
               'ne':'Segmented neutrophiles',\
               'eo':'Eosinophiles',\
               'ba':'Basophiles',\
               'mo':'Monocytes',\
               'ly':'Lymphocytes',\
               'lgly':'Large granular lymphocytes',\
               'rly':'Reactive lymphocytes'}


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
# TODO: add 25 as parameter
    
        if save_diag:
            fig = plt.figure(dpi=300)
            # Plot manual
            fig=imtools.plotShapes(im,annotations_bb,color='b',\
                                   detect_shapes=list(wbc_types.keys()),text='ALL',fig=fig)
            # Plot automatic
            fig=imtools.plotShapes(im,shapelist,color='r',\
                                   detect_shapes='ALL',text=('WBC'),fig=fig)
            head, tail = str.split(xml_file_2,'.')
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
            
            n_wbc=0
            for each_bb in annotations_bb:
                if each_bb[0] in list(wbc_types.keys()):
                    n_wbc+=1
                    
            n_wbc_detected=0
            n_wbc_matched=0
            for each_shape in shapelist:
                if each_shape[0]=='WBC':
                    n_wbc_detected+=1;
                    for each_bb in annotations_bb:
                        if each_bb[0] in list(wbc_types.keys()):
                            bb=Path(each_bb[2])
                            intersect = bb.contains_points(each_shape[2])    
                            if intersect.sum()>0:
                                p_over=intersect.sum()/len(each_shape[2])
                                n_wbc_matched+=p_over
                                annotations_bb.remove(each_bb)
                                break
                            
            detect_stat.append((image_file,n_wbc,n_wbc_detected,n_wbc_matched))    
 
    n_images=len(image_list_indir)
    n_wbc=[]
    n_wbc_detected=[]
    n_wbc_matched=[]    
    for stats in detect_stat:
        n_wbc.append(stats[1])
        n_wbc_detected.append(stats[2])
        n_wbc_matched.append(stats[3])
        
    print('images in dir:'+str(n_images))
    print('images with manual and automatic annotations:'+str(len(detect_stat)))
    print('n wbc total:'+str(sum(n_wbc)))
    print('n wbc detected total:'+str(sum(n_wbc_detected)))
    print('n wbc matched total:'+str(sum(n_wbc_matched)))
    if save_diag:
        with open(os.path.join(output_dir,'eval_stats.txt'), 'w') as eval_file:
            eval_file.write('images in dir:         {0:.0f}\n'.format(n_images))
            eval_file.write('images with manual and automatic annotations: {0:.0f}\n'.format(len(detect_stat)))
            eval_file.write('n wbc total:           {0:.0f}\n'.format(sum(n_wbc)))
            eval_file.write('n wbc detected total:  {0:.0f}\n'.format(sum(n_wbc_detected)))
            eval_file.write('n wbc matched total:   {0:.0f}\n'.format(sum(n_wbc_matched)))

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