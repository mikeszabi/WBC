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
from mpl_toolkits.mplot3d import Axes3D

import random

def segment_fg_bg_sv_kmeans4(csp, init_centers, vis_diag=False):  
    # segmentation on csp image
    
    #param=cfg.param()
    # TODO: use parameters from cfg
    
    # KMEANS on saturation and intensity
    Z = csp.reshape((-1,3))
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

    lab_all=label.reshape((csp.shape[0:2]))    
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

def segment_cell_hs_kmeans3(csp, mask, vis_diag=False):  
    Z = csp.reshape((-1,3))/255
    Z = np.float32(Z)
                  
    # mask with overexpo
    Z_mask=mask.reshape((-1,1))>0
    Z_mask=Z_mask.flatten()

    # select all channels
    Z_1=Z[Z_mask>0,0:2]
    Z=Z[:,0:2]

    kmeans = KMeans(n_clusters=3, random_state=0).fit(Z_1)
    
    # TODO: initialize centers from histogram peaks
    center = kmeans.cluster_centers_
    label = kmeans.labels_
    #print(center)
    colors = cm.jet(np.linspace(1/(label.max()+1), 1, label.max()+1))


    if vis_diag:
        rs=random.sample(range(0, Z_1.shape[0]-1), 1000)
        Z_rs=Z_1[rs,:]
        fig = plt.figure("scatter", figsize=(4, 3))
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
       
    # adding meaningful labels
    lab=np.zeros(lab_ok.shape).astype('uint8')
    
    #ind_hue=np.argsort(center[:,0])
    ind_sat=np.argsort(center[:,1])
   
    sure_ind=[]
    # wbc and cell membrane - largest saturation
    lab[lab_ok==ind_sat[-1]]=1
    sure_ind.append(ind_sat[-1])
                 
    # sure foreground mask -largest saturation
    #if ind_sat[-1] not in sure_ind:
    #if ind_val[ind_sat[-1]]>ind_val[ind_sat[-2]]:        
    lab[lab_ok==ind_sat[-2]]=3
    sure_ind.append(ind_sat[-2])
    #else:
    lab[lab_ok==ind_sat[-3]]=2
    sure_ind.append(ind_sat[-3])
       
    # lab == 0 : background
    # lab == 1 : rbc cell membrane, wbc
    # lab ==2 : unknown 
    # lab == 3 : rbc

     
    return center, lab

def segment_wbc_2(csp, mask, vis_diag=False):  
    Z = csp.reshape((-1,3))
    Z = np.float32(Z)
                  
    # mask with overexpo
    Z_mask=mask.reshape((-1,1))>0
    Z_mask=Z_mask.flatten()

    # select all channels
    Z_1=Z[Z_mask>0,0:3]
    Z=Z[:,0:3]

    kmeans = KMeans(n_clusters=3, random_state=0).fit(Z_1)
    
    # TODO: initialize centers from histogram peaks
    center = kmeans.cluster_centers_
    label = kmeans.labels_
    #print(center)

    if vis_diag:
        rs=random.sample(range(0, Z_1.shape[0]-1), 1000)
        Z_rs=Z_1[rs,:]
        label_rs=label[rs]

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()

        ax.scatter(Z_rs[:, 0], Z_rs[:, 1], Z_rs[:, 2], c=label_rs.astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Hue')
        ax.set_ylabel('Saturation')
        ax.set_zlabel('Value')
        plt.show()
    
        plt.show()


    lab_all=np.zeros(Z.shape[0])
    lab_all[Z_mask==0]=-1
    lab_all.flat[Z_mask>0]=label
    imtools.normalize(lab_all.reshape((csp.shape[0:2])),vis_diag=vis_diag,fig='wbc_labels')
            
    return center, lab_all