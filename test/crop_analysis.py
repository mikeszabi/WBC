# -*- coding: utf-8 -*-
"""
Created on Tue May 30 06:40:14 2017

@author: picturio
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib qt5

df=pd.read_csv(r'c:\Users\picturio\OneDrive\WBC\DATA\Detected_Cropped\detections.csv',delimiter=';')

np.unique(df['wbc'])

df_wbc=df[df.wbc=='un']


r = df_wbc['radius'].apply(lambda x: float(x.replace(' ','').replace('[','').split('.')[0]))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(10,150)
r.hist(bins=30,ax=ax)
