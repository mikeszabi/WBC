# -*- codinf,0g: utf-8 -*-
"""
Created on Mon Jan 23 20:40:00 2017

@author: SzMike
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
%matplotlib qt5



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = Z[:,0]
ys = Z[:,1]
zs = Z[:,2]
ax.scatter(xs, ys, zs)

ax.set_xlabel('blue')
ax.set_ylabel('green')
ax.set_zlabel('Z Labelred')

plt.show()

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = Z[:,0]
    ys = Z[:,1]
    zs = Z[:,2]
    ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()