# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:27:39 2017

@author: Szabolcs
"""
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import color
from scipy import ndimage
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


import imtools
import blobtools
from scipy import ndimage
from scipy import signal
from skimage import color

fill_tsh=0.75
fig='Blob'
scale=0.4
##

mask_fg=label_fg_bg>30

label_im, nb_labels = ndimage.label(mask_fg)
sizes = np.sqrt(ndimage.sum(mask_fg, label_im, range(nb_labels + 1)))/np.pi/scale
stop_hist, bin_edges=np.histogram(sizes,50)
fi=plt.figure(fig+'histogram of max radii')
ax=fi.add_subplot(111)
ax.plot(bin_edges[1:],stop_hist)
#    mask_fg=label_fg_bg>30
#    mask_fg=morphology.binary_closing(mask_fg,morphology.disk(1))
#    mask_fg=morphology.binary_opening(mask_fg,morphology.disk(1))    
#    
    
#    label_cc = measure.label(morphology.binary_erosion(mask_fg,morphology.disk(1)))
max_dim=max(mask_fg.shape)

min_r=int(max_dim/scale/100)
max_r=int(max_dim/scale/30)

#    mask_fg=morphology.remove_small_holes(mask_fg>0, 
#                                          min_size=scale*min_r*min_r*np.pi, 
#                                          connectivity=2)

# TODO add these to parameters
r_list = np.linspace(min_r, max_r, (max_r-min_r)+1)
start_r=0
r_list=r_list[r_list>start_r]
     
mask = (mask_fg.copy()).astype('float64')

im_filtered = [ndimage.convolve(mask, morphology.disk(r*scale))/(morphology.disk(r*scale).sum()) for r in r_list]

fill_cube = np.dstack(im_filtered)
delta_cube = fill_cube[:,:,1:]-fill_cube[:,:,0:-1]

inc_cube=np.cumsum(delta_cube>=0, axis=2)

label_inc_stop=np.zeros(mask_fg.shape)
fill_at_max=np.zeros(mask_fg.shape)

for i in range(delta_cube.shape[2]):
    label_inc_stop[inc_cube[:,:,i]==i]=i
    fill_at_max[inc_cube[:,:,i]==i]=fill_cube[inc_cube[:,:,i]==i,i]          

label_inc_stop[fill_at_max<fill_tsh]=0

threshold=0
# TODO : make fp dependent on rbc size
fp=int(max_dim/scale/100)
local_maxima_inc = feature.peak_local_max(inc_cube, threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((fp,fp,3)),
                                      threshold_rel=0.0,
                                      exclude_border=False)

threshold=fill_tsh
fp=3 #int(max_dim/scale/150)   
local_maxima_fill = feature.peak_local_max(fill_cube[:,:,1:], threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((fp,fp,1)),
                                      threshold_rel=0.0,
                                      exclude_border=False)


markers_r=np.zeros(mask_fg.shape)
for i in range(inc_cube.shape[2]):
    tmp_mask=np.logical_and(local_maxima_inc[:,:,i]>0,local_maxima_fill[:,:,i]>0)
    #tmp_mask=morphology.binary_dilation(tmp_mask,morphology.disk(3))
    markers_r[tmp_mask]=r_list[i+1]
markers_r_raw=markers_r.copy()        


# estimate RBC size
stops=label_inc_stop[markers_r>0].flatten()

stop_hist, bin_edges=np.histogram(stops,int(len(r_list)/2))

# finding local maxima in histogram
i1=signal.argrelmax(stop_hist)
i2=np.argmax(stop_hist[i1])
i_max=i1[0][i2]

rbc_ind=np.round(bin_edges[i_max+1]).astype('int64')
   
rbcR=r_list[rbc_ind]

# final markers
markers_r[np.logical_or(markers_r<rbcR*0.75,markers_r>rbcR*1.5)]=0

# TODO create marker list with corresponding radius

markers=markers_r>0  
markers=morphology.binary_dilation(markers_r,morphology.disk(int(rbcR*scale/4))).astype('uint8')
#markers=morphology.binary_opening(markers,morphology.disk(1)).astype('uint8')
# TODO create statistics on spatial distribution of markers - add to diagnostics
markers[markers_r>0]=markers_r[markers_r>0]


fi=plt.figure(fig+'histogram of max radii')
ax=fi.add_subplot(111)
ax.plot(r_list[np.round(bin_edges[1:]).astype('int64')],stop_hist)
# TODO check sharpness of histogram maxima - add diagnostics problem when wide
blob=imtools.overlayImage(color.gray2rgb(255*mask).astype('uint8'),markers_r_raw>0,(0,1,0),1,vis_diag=False,fig=fig+'blob_mask')            
blob=imtools.overlayImage(blob,markers>0,(1,0,0),1,vis_diag=True,fig=fig+'blob_mask')
