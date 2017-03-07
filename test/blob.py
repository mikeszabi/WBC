# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:27:39 2017

@author: Szabolcs
"""
from scipy import ndimage
from scipy import signal

fill_tsh=0.75

##

mask_fg_clear=label_fg_bg>30

mask_fg_clear=label_wbc>0

min_r=int(max(mask_fg_clear.shape)/scale/100)
max_r=int(max(mask_fg_clear.shape)/scale/30)
r_list = np.linspace(min_r, max_r, (max_r-min_r)+1)
         
image = (mask_fg_clear).astype('float64')

im_filtered = [ndimage.convolve(image, morphology.disk(r*scale))/(morphology.disk(r*scale).sum()) for r in r_list]

fill_cube = np.dstack(im_filtered)
delta_cube = fill_cube[:,:,1:]-fill_cube[:,:,0:-1]

inc_cube=np.cumsum(delta_cube>=0, axis=2)

label_inc_stop=np.zeros(mask_fg_clear.shape)
fill_at_max=np.zeros(mask_fg_clear.shape)

for i in range(delta_cube.shape[2]):
    label_inc_stop[inc_cube[:,:,i]==i]=i
    fill_at_max[inc_cube[:,:,i]==i]=fill_cube[inc_cube[:,:,i]==i,i]          


label_inc_stop[fill_at_max<fill_tsh]=0

threshold=0
fp=int(max(mask_fg_clear.shape)/50)
local_maxima_1 = feature.peak_local_max(label_inc_stop, threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((fp,fp)),
                                      threshold_rel=0.0,
                                      exclude_border=False)

threshold=fill_tsh
local_maxima_2 = feature.peak_local_max(fill_cube, threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((3,3,1)),
                                      threshold_rel=0.0,
                                      exclude_border=False)


markers_1 = measure.label(local_maxima_1,return_num=False)
markers_2=measure.label(np.sum(local_maxima_2,axis=2),return_num=False)  
markers_2 = morphology.binary_opening(markers_2,morphology.disk(1))

markers=np.logical_and(markers_1>0,markers_2>0)

# estinate RBC size
stops=label_inc_stop[markers>0].flatten()

stop_hist, bin_edges=np.histogram(stops,20)

# finding local maxima in histogram
i1=signal.argrelmax(stop_hist)
i2=np.argmax(stop_hist[i1])
i_max=i1[0][i2]

rbc_ind=np.round(bin_edges[i_max+1]).astype('int64')
   
rbcR=r_list[rbc_ind]

# final markers
markers_3=markers.copy()
#markers_3[np.logical_or(label_inc_stop<rbc_ind*0.5,label_inc_stop>rbc_ind*1.5)]=0
markers_3[label_inc_stop<np.argmax(r_list>param.rbcR*0.75)]=0
markers_3=morphology.binary_dilation(markers_3,morphology.disk(3))


if vis_diag:
    fi=plt.figure('histogram of max radii')
    ax=fi.add_subplot(111)
    ax.plot(r_list[np.round(bin_edges[1:]).astype('int64')],stop_hist)
    
    rbc_blob=imtools.overlayImage(im_resize,markers_2>0,(0,1,0),1,vis_diag=True,fig='rbc_mask')
    rbc_blob=imtools.overlayImage(rbc_blob,markers_3,(1,0,0),1,vis_diag=True,fig='rbc_mask')
    rbc_blob=imtools.overlayImage(rbc_blob,markers_1>0,(0,0,1),1,vis_diag=True,fig='blob_mask')

