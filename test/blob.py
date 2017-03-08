# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:27:39 2017

@author: Szabolcs
"""
from scipy import ndimage
from scipy import signal
from skimage import color

fill_tsh=0.75

##

#mask_fg=label_fg_bg>30

mask_fg=label_wbc>0
mask_fg=morphology.binary_closing(mask_fg,morphology.disk(2))
mask_fg=morphology.binary_opening(mask_fg,morphology.disk(2))

min_r=int(max(mask_fg.shape)/scale/100)
max_r=int(max(mask_fg.shape)/scale/30)
r_list = np.linspace(min_r, max_r, (max_r-min_r)+1)
   
start_r=0
r_list=r_list[r_list>start_r]
      
mask = (mask_fg).astype('float64')

im_filtered = [ndimage.convolve(mask, morphology.disk(r*scale))/(morphology.disk(r*scale).sum()) for r in r_list]

fill_cube = np.dstack(im_filtered)
delta_cube = fill_cube[:,:,1:]-fill_cube[:,:,0:-1]

inc_cube=np.cumsum(delta_cube>=0, axis=2)

label_inc_stop=np.zeros(mask_fg.shape)
#fill_at_max=np.zeros(mask_fg.shape)

for i in range(delta_cube.shape[2]):
    label_inc_stop[inc_cube[:,:,i]==i]=i
    #fill_at_max[inc_cube[:,:,i]==i]=fill_cube[inc_cube[:,:,i]==i,i]          


label_inc_stop[fill_at_max<fill_tsh]=0

threshold=0
fp=int(max(mask_fg.shape)/50)
local_maxima_inc = feature.peak_local_max(inc_cube, threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((fp,fp,3)),
                                      threshold_rel=0.0,
                                      exclude_border=False)

threshold=fill_tsh
fp=int(max(mask_fg.shape)/50)

local_maxima_fill = feature.peak_local_max(fill_cube[:,:,1:], threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((fp,fp,3)),
                                      threshold_rel=0.0,
                                      exclude_border=False)

markers=np.zeros(mask_fg.shape)
for i in range(inc_cube.shape[2]):
    tmp_mask=np.logical_and(local_maxima_inc[:,:,i]>0,local_maxima_fill[:,:,i]>0)
    #tmp_mask=morphology.binary_dilation(tmp_mask,morphology.disk(3))
    markers[tmp_mask]=1

blob=imtools.overlayImage(color.gray2rgb(255*mask).astype('uint8'),markers>0,(0,1,0),1,vis_diag=True,fig='blob_mask')


#markers_1 = measure.label(local_maxima_1,return_num=False)
#markers_2 = measure.label(local_maxima_2,return_num=False)
#
##markers_2=measure.label(np.sum(local_maxima_2,axis=2),return_num=False)  
#markers_2 = morphology.binary_opening(markers_2,morphology.disk(2))

#markers=np.logical_and(markers_1>0,markers_2>0)

# estinate RBC size
"""
stops=label_inc_stop[markers>0].flatten()

stop_hist, bin_edges=np.histogram(stops,20)

stop_hist[i_max:].sum()/stop_hist.sum()


# finding local maxima in histogram
i1=signal.argrelmax(stop_hist)
i2=np.argmax(stop_hist[i1])
i_max=i1[0][i2]

rbc_ind=np.round(bin_edges[i_max+1]).astype('int64')
   
rbcR=r_list[rbc_ind]
"""
# final markers
markers_3=markers.copy()
#markers_3[np.logical_or(label_inc_stop<rbc_ind*0.1,label_inc_stop>rbc_ind*2)]=0
#markers_3[label_inc_stop<np.argmax(r_list>rbcR*0.5)]=0
markers_3=morphology.binary_dilation(markers_3,morphology.disk(3))
 
blob=imtools.overlayImage(color.gray2rgb(255*mask).astype('uint8'),markers_3>0,(1,0,0),1,vis_diag=True,fig='blob_mask')
#blob=imtools.overlayImage(blob,255*(markers_3>0),(1,0,0),1,vis_diag=True,fig='blob_mask')
    
#TODO: count number of oversized rbc

if vis_diag:
    fi=plt.figure('histogram of max radii')
    ax=fi.add_subplot(111)
    ax.plot(r_list[np.round(bin_edges[1:]).astype('int64')],stop_hist)
    
    rbc_blob=imtools.overlayImage(im_resize,markers_1>0,(0,1,0),1,vis_diag=True,fig='rbc_mask')
    rbc_blob=imtools.overlayImage(rbc_blob,markers_3,(1,0,0),1,vis_diag=True,fig='rbc_mask')
    rbc_blob=imtools.overlayImage(rbc_blob,markers_1>0,(0,0,1),1,vis_diag=True,fig='blob_mask')

     blob=imtools.overlayImage(color.gray2rgb(255*mask).astype('uint8'),markers_3>0,(0,1,0),1,vis_diag=True,fig='blob_mask')
     blob=imtools.overlayImage(blob,255*(markers_3>0),(1,0,0),1,vis_diag=False,fig='blob_mask')
    
    
    blob=imtools.overlayImage(blob,markers_1>0,(0,0,1),1,vis_diag=True,fig='blob_mask')
      
