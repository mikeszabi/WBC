# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys
import skimage.io as io
io.use_plugin('pil') # Use only the capability of PIL
import matplotlib.pyplot as plt
from matplotlib.path import Path
import argparse
import numpy as np
import pandas as pd

import annotations
import imtools
import cfg

def evaluate_wbc_detection(image_dir,output_dir,save_diag=False):
    
    plt.ioff()

    param=cfg.param()
    wbc_types=param.wbc_types
    wbc_basic_types=param.wbc_basic_types
    wbc_type_dict=param.wbc_type_dict

    image_list_indir=imtools.imagelist_in_depth(image_dir,level=1)
    print('processing '+str(len(image_list_indir))+' images')

    anns=[]   
    dets=[]  
    nOK=0
    
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
                continue
        else:
            continue
         
        # keep WBC detections    
        annotations_bb = [bb for bb in annotations_bb if bb[0] in list(wbc_types.keys())]
        """
        READ auto annotations
        """            
        file_name=os.path.basename(image_file)
        head, tail=os.path.splitext(file_name)
        xml_file_2=os.path.join(output_dir,head+'.xml')
        if os.path.isfile(xml_file_2):
            try:
                xmlReader = annotations.AnnotationReader(xml_file_2)
                detections_bb=xmlReader.getShapes()
            except:
                continue
        else:
            continue
        # keep WBC detections    
        detections_bb = [bb for bb in detections_bb if bb[0] in list(wbc_basic_types.keys())]

                
        """
        READ image
        """
        try:
            im = io.imread(image_file) # read uint8 image
        except:
            continue
        
        nOK+=1
        
        """
        REMOVE ANNOTATIONS CLOSE TO BORDER
        """
        for each_bb in annotations_bb:
            bb=each_bb[2]
            if min((im.shape[1],im.shape[0])-np.average(bb,axis=0))<param.border or min(np.average(bb,axis=0))<param.border:
                annotations_bb.remove(each_bb)
        """
        REMOVE ANNOTATIONS CLOSE TO BORDER
        """
        for each_bb in detections_bb:
            bb=each_bb[2]
            if min((im.shape[1],im.shape[0])-np.average(bb,axis=0))<param.border or min(np.average(bb,axis=0))<param.border:
                detections_bb.remove(each_bb)
# TODO: add 25 as parameter
    
        if save_diag and (detections_bb):
            fig = plt.figure('wbc_annotations',figsize=(20,20),dpi=300)
            # Plot manual
            fig=imtools.plotShapes(im,annotations_bb,color='b',marker='x',ha='left',va='top',\
                                   detect_shapes=list(wbc_types.keys()),text=list(wbc_types.keys()),fig=fig)
            # Plot automatic
            fig=imtools.plotShapes(im,detections_bb,color='r',ha='right',va='bottom',\
                                   detect_shapes='ALL',text=list(wbc_basic_types.keys()),fig=fig)
            head, tail = str.rsplit(os.path.abspath(xml_file_2),'.',1)
            detect_image_file=os.path.join(head+'_wbc_annotations.jpg')
            fig.savefig(detect_image_file,dpi=300)
            plt.close(fig)
        
        """
        COMPARE manual vs. automatic detections
        """
#        x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
#        x, y = x.flatten(), y.flatten()
#        points = np.vstack((x,y)).T
                 
        match_indices=[]                       
        for i,each_bb in enumerate(annotations_bb):   
            for j,each_shape in enumerate(detections_bb):  
                 bb=Path(each_bb[2])
                 center_point=[np.mean(each_shape[2],axis=0).tolist()]
                 intersect = bb.contains_points(center_point)    
                 if intersect.sum()>0:
                     match_indices.append((i,j))
        for mi in match_indices:
            anns.append(wbc_type_dict[annotations_bb[mi[0]][0]])
            dets.append(detections_bb[mi[1]][0])
        for ai in list(set(range(len(annotations_bb)))-set([i[0] for i in  match_indices])):
            anns.append(wbc_type_dict[annotations_bb[ai][0]])
            dets.append('ND')
            print(annotations_bb[ai][0])
        for di in list(set(range(len(detections_bb)))-set([i[1] for i in  match_indices])):
            anns.append('NA')
            dets.append(detections_bb[di][0])
        

    """
    AGGREGATE STATISTICS
    """ 
    tmp_list=list(wbc_basic_types.keys())
    tmp_list.append('NA')
    ann_cats = pd.Categorical(anns, categories=tmp_list)
    tmp_list=list(wbc_basic_types.keys())
    tmp_list.append('ND')
    det_cats = pd.Categorical(dets, categories=tmp_list)
    contingency=pd.crosstab(ann_cats, det_cats)  
    print(contingency)
    # rows are manual annotations, cols are automatic annotations
    
    """
    WRITE TO CSV
    """
    contingency.to_csv(os.path.join(output_dir,'eval_contingency_'+os.path.basename(image_dir)+'.csv'))
    print('n images evaluated:\t'+str(len(image_list_indir)))
    print('n images ok:\t\t'+str(nOK))


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