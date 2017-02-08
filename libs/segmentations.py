# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:33:32 2017

@author: SzMike
"""

import numpy as np;
import imtools
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import matplotlib.cm as cm

import random

def segment_fg_bg_sv_kmeans4(hsv, init_centers, vis_diag=False):  
    # segmentation on hsv image
    
    #param=cfg.param()
    # TODO: use parameters from cfg
    
    # KMEANS on saturation and intensity
    Z = hsv.reshape((-1,3))
    Z = np.float32(Z)/256

 
    # select saturation and value channels
    Z=Z[:,1:3]

    kmeans = KMeans(n_clusters=4, random_state=1, init=init_centers).fit(Z)
    
    # TODO: initialize centers from histogram peaks
    center = np.uint8(kmeans.cluster_centers_*256)
    label = kmeans.labels_
    #print(center)
    colors = cm.jet(np.linspace(0, 1, label.max()+1))

    if vis_diag:
        rs=random.sample(range(0, Z.shape[0]-1), 1000)
        Z_rs=Z[rs,:]

        fig = plt.figure("scatter", figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111)
        plt.cla()
        label_rs = label[rs]
        ax.set_xlabel('Saturation')
        ax.set_ylabel('Value')
        for i, c in enumerate(center):
            ax.scatter(Z_rs[label_rs==i, 0], Z_rs[label_rs==i, 1], color=colors[i,:])
            ax.plot(c[0]/255, c[1]/255, 'o',markerfacecolor='k', markeredgecolor='k', markersize=15)

        plt.show()

    lab_all=label.reshape((hsv.shape[0:2]))    
    if vis_diag:    
        imtools.normalize(lab_all,vis_diag=vis_diag,fig='fg_bg_labels')
    
    # adding meaningful labels
    lab=2*np.ones(lab_all.shape).astype('uint8')
    
    ind_sat=np.argsort(center[:,0])
    ind_val=np.argsort(center[:,1])
   
    sure_ind=[]
    # sure background mask - largest intensity
    lab[lab_all==ind_val[-1]]=1
    sure_ind.append(ind_val[-1])
                 
    # sure foreground mask -largest saturation
    #if ind_sat[-1] not in sure_ind:
    #if ind_val[ind_sat[-1]]>ind_val[ind_sat[-2]]:        
    lab[lab_all==ind_sat[-1]]=3
    sure_ind.append(ind_sat[-1])
    #else:
    lab[lab_all==ind_sat[-2]]=3
    sure_ind.append(ind_sat[-2])
       

    # lab == 1 : sure bckg
    # lab ==2 : unsure
    # lab == 3 : sure foreground

    return center, lab