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

def segment_fg_bg_sv_kmeans(csp, init_centers, n_clusters=4, vis_diag=False):  
    # segmentation on csp image
    
    #param=cfg.param()
    # TODO: use parameters from cfg
    
    # KMEANS on saturation and intensity
    Z = csp.reshape((-1,3))
    Z = np.float32(Z)

 
    # select saturation and value channels
    Z=Z[:,1:3]

    kmeans = KMeans(n_clusters=n_clusters, random_state=1, init=init_centers).fit(Z)
    
    # TODO: initialize centers from histogram peaks
    center = kmeans.cluster_centers_
    label = kmeans.labels_
    #print(center)
    colors = cm.jet(np.linspace(0, 1, label.max()+1))

    if vis_diag:
        rs=random.sample(range(0, Z.shape[0]-1), 1000)
        Z_rs=Z[rs,:]

        fig = plt.figure("scatter_bg", figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111)
        plt.cla()
        label_rs = label[rs]
        ax.set_xlabel('Saturation')
        ax.set_ylabel('Value')
        for i, c in enumerate(center):
            ax.scatter(Z_rs[label_rs==i, 0], Z_rs[label_rs==i, 1], color=colors[i,:])
            ax.plot(c[0], c[1], 'o',markerfacecolor='k', markeredgecolor='k', markersize=15)

        plt.show()

    lab_all=label.reshape((csp.shape[0:2]))    
    if vis_diag:    
        imtools.normalize(lab_all,vis_diag=vis_diag,fig='fg_bg_labels')
    
    # adding meaningful labels
#    lab=2*np.ones(lab_all.shape).astype('uint8')
#    
#    ind_sat=np.argsort(center[:,0])
#    ind_val=np.argsort(center[:,1])
#   
#    sure_ind=[]
#    # sure background mask - largest intensity
#    lab[lab_all==ind_val[-1]]=1
#    sure_ind.append(ind_val[-1])
#                 
#    # sure foreground mask -largest saturation
#    #if ind_sat[-1] not in sure_ind:
#    #if ind_val[ind_sat[-1]]>ind_val[ind_sat[-2]]:        
#    lab[lab_all==ind_sat[-1]]=3
#    sure_ind.append(ind_sat[-1])
#    #else:
#    lab[lab_all==ind_sat[-2]]=3
#    sure_ind.append(ind_sat[-2])
#       

    # lab == 1 : sure bckg
    # lab ==2 : unsure
    # lab == 3 : sure foreground

    return center, lab_all

def segment_cell_hs_kmeans(csp, mask, cut_channel=1, n_clusters=5, vis_diag=False):  
    rgb_range=((330/360*255,30/360*255), (75/360*255,135/360*255), (180/360*255,270/360*255))

    cut = np.mean(rgb_range[cut_channel])
    
    
    Z = csp.reshape((-1,3))
    Z = np.float32(Z)
                  
    # mask with overexpo
    Z_mask=mask.reshape((-1,1))>0
    Z_mask=Z_mask.flatten()

    # select all channels
    Z=Z[:,0:2]
    Z[Z[:,0]<cut,0]=Z[Z[:,0]<cut,0]+cut
    Z_1=Z[Z_mask>0,0:2]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Z_1)
    
    # TODO: initialize centers from histogram peaks
    center = kmeans.cluster_centers_
    label = kmeans.labels_
    #print(center)
    
#    Q95=np.zeros(center.shape[0])
#    for l in np.unique(label):
#        sats=Z_1[label==l,1]
#        hist, bin_edges = np.histogram(sats, range(256), normed=True)
#        cdf = np.cumsum(hist)
#        Q95[l]=np.argwhere(cdf>0.95)[0,0]
#        
    colors = cm.jet(np.linspace(1/(label.max()+1), 1, label.max()+1))


    if vis_diag:
        rs=random.sample(range(0, Z_1.shape[0]-1), 1000)
        Z_rs=Z_1[rs,:]
        fig = plt.figure("scatter_cell", figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111)
        plt.cla()
        label_rs = label[rs]
        ax.set_xlabel('Hue')
        ax.set_ylabel('Saturation')
        for i, c in enumerate(center):
            ax.scatter(Z_rs[label_rs==i, 0], Z_rs[label_rs==i, 1], color=colors[i,:])
            ax.plot(c[0], c[1], 'o',markerfacecolor='k', markeredgecolor='k', markersize=15)

        plt.show()

    lab_all=np.zeros(Z.shape[0])
    lab_all[Z_mask==0]=-1
    lab_all.flat[Z_mask>0]=label
    lab_ok=lab_all.reshape((csp.shape[0:2]))
    if vis_diag:
        imtools.normalize(lab_ok,vis_diag=vis_diag,fig='wbc_labels')
  
    return center, lab_ok


def segment_wbc_hue(csp, mask, cut_channel=1, n_clusters=2, vis_diag=False):  
    rgb_range=((330/360*255,30/360*255), (75/360*255,135/360*255), (180/360*255,270/360*255))

    cut = np.mean(rgb_range[cut_channel])
    
    Z = csp.reshape((-1,3))
    Z = np.float32(Z)
                  
    # mask with overexpo
    Z_mask=mask.reshape((-1,1))>0
    Z_mask=Z_mask.flatten()

    # select all channels
    Z=Z[:,0:1]
    Z[Z<cut]+=cut
    Z_1=Z[Z_mask>0]
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Z_1)
    
    # TODO: initialize centers from histogram peaks
    center = kmeans.cluster_centers_
    label = kmeans.labels_
    #print(center)
    
    colors = cm.jet(np.linspace(1/(label.max()+1), 1, label.max()+1))

    if vis_diag:
        rs=random.sample(range(0, Z_1.shape[0]-1), 1000)
        Z_rs=Z_1[rs,:]
        fig = plt.figure("scatter_cell", figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111)
        plt.cla()
        label_rs = label[rs]
        ax.set_xlabel('Hue')
        ax.set_ylabel('Saturation')
        for i, c in enumerate(center):
            ax.scatter(Z_rs[label_rs==i, 0], Z_rs[label_rs==i, 0], color=colors[i,:])
            ax.plot(c[0], c[0], 'o',markerfacecolor='k', markeredgecolor='k', markersize=15)       
        plt.show()

    lab_all=np.zeros(Z.shape[0])
    lab_all[Z_mask==0]=-1
    lab_all.flat[Z_mask>0]=label
    
    lab_ok=lab_all.reshape((csp.shape[0:2]))
    if vis_diag:
        imtools.normalize(lab_ok,vis_diag=vis_diag,fig='wbc_labels')
  
    return center, lab_ok