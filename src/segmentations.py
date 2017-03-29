# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:33:32 2017

@author: SzMike
"""
import warnings
import numpy as np;
import imtools
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random


def segment_hsv(csp, mask=None, init_centers='k-means++', cut_channel=1, chs=(0,1,2), n_clusters=3, vis_diag=False): 
    
    ch_names=['Hue','Saturation','Value']
    rgb_range_in_hue=((-30/360,30/360), (75/360,135/360), (180/360,240/360))
    
    if mask is None:
        mask=np.ones(csp.shape[0:2])
    
    Z = csp.reshape((-1,3))
    Z = np.float32(Z)
                  
    Z_mask=mask.reshape((-1,1))>0
    Z_mask=Z_mask.flatten()

    # select all channels
    Z=Z[:,chs]
    for i, c in enumerate(chs):
        if c==0:
            cut = 255*np.mean(rgb_range_in_hue[cut_channel])   
            Z[Z[:,i]<cut,i]=Z[Z[:,i]<cut,i]+cut
    Z_1=Z[Z_mask>0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, init=init_centers).fit(Z_1)
    
    center = kmeans.cluster_centers_
    label = kmeans.labels_
    #print(center)
    if (Z_mask==0).sum()>0:
        colors = cm.jet(np.linspace(1/(label.max()+2), 1, label.max()+1))
    else:
        colors = cm.jet(np.linspace(0, 1, label.max()+1))

    if vis_diag:
        rs=random.sample(range(0, Z_1.shape[0]-1), min(1000,Z_1.shape[0]-1))
        Z_rs=Z_1[rs,:]
        fig = plt.figure("scatter", figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()
        label_rs = label[rs]

        ax.set_xlabel(ch_names[chs[0]])
        ax.set_ylabel(ch_names[chs[1]])
        ax.set_zlabel(ch_names[chs[2]])    
        for i, c in enumerate(center):
            ax.scatter(Z_rs[label_rs==i, 0], Z_rs[label_rs==i, 1], Z_rs[label_rs==i, 2], color=colors[i,:])                      
            ax.scatter(c[0],c[1],c[2], 'o', s=100, c='k')
            ax.text(c[0],c[1],c[2],str(i),bbox=dict(facecolor='white', alpha=1),size='x-large',va='top',weight='heavy')
        plt.show()

    lab_all=np.zeros(Z.shape[0])
    lab_all[Z_mask==0]=-1
    lab_all.flat[Z_mask>0]=label
    lab_ok=lab_all.reshape((csp.shape[0:2]))
    
    if vis_diag:
        imtools.normalize(lab_ok,vis_diag=vis_diag,fig='labels')
   
    return center, lab_ok

def center_diff_matrix(centers,metric='euclidean'):    
    return pairwise_distances(centers,metric='euclidean')