# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 21:15:05 2017

@author: SzMike
"""

from sklearn.cluster import KMeans

center,label,inertia=KMeans(n_clusters=3, random_state=0).fit(Z_1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets

Z = hsv.reshape((-1,3))
Z = np.float32(Z)/256
              
# mask with overexpo
Z_mask=overexpo_mask.reshape((-1,1))==0
Z_mask=Z_mask.flatten()

# 3d clustering
Z_1=Z[Z_mask,0:3]
rs=random.sample(range(0, Z.shape[0]-1), 1000)
Z_1=Z_1[rs,:]


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
est=KMeans(n_clusters=3).fit(Z_1)
labels = est.labels_
ax.scatter(Z_1[:, 0], Z_1[:, 1], Z_1[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Hue')
ax.set_ylabel('Saturation')
ax.set_zlabel('Value')
plt.show()

#2d (saturation, value) clustering
Z_1=Z[Z_mask,1:3]
rs=random.sample(range(0, Z.shape[0]-1), 1000)
Z_1=Z_1[rs,:]

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = fig.add_subplot(111)
plt.cla()
est=KMeans(n_clusters=3).fit(Z_1)
labels = est.labels_
ax.scatter(Z_1[:, 0], Z_1[:, 1], c=labels.astype(np.float))

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_xlabel('Saturation')
ax.set_ylabel('Value')
plt.show()

#
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #K = 3
    # TODO: why 3 ?
    #ret,label,center=cv2.kmeans(Z_1,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    #center = np.uint8(center*256)
    