# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:36:12 2017

@author: SzMike
"""

mask_fg=label_wbc>0
mask_fg=morphology.binary_closing(mask_fg,morphology.disk(2))
mask_fg=morphology.binary_opening(mask_fg,morphology.disk(2))

min_r=int(max(mask_fg.shape)/scale/50)
max_r=int(max(mask_fg.shape)/scale/20)
r_list = np.linspace(min_r, max_r, (max_r-min_r)+1)
   
start_r=0
r_list=r_list[r_list>start_r]
      
mask = (mask_fg).astype('float64')

im_filtered = [ndimage.convolve(mask, morphology.disk(r*scale))/(morphology.disk(r*scale).sum()) for r in r_list]

fill_cube = np.dstack(im_filtered)

fp=int(max_dim/scale/50)   
threshold=0.25
local_maxima_fill = feature.peak_local_max(fill_cube, threshold_abs=threshold,
                                      indices=True,
                                      footprint=np.ones((fp,fp,3)),
                                      threshold_rel=0.0,
                                      exclude_border=False)


fig=plt.figure('circle image')
axs=fig.add_subplot(111)
axs.imshow(im_resize)  
for l in local_maxima_fill:
    circ=plt.Circle((l[1],l[0]), radius=r_list[l[2]]*scale, color='g', fill=False)
    axs.add_patch(circ)
    
    
markers_r=np.zeros(mask_fg.shape)
for i in range(inc_cube.shape[2]):
    tmp_mask=local_maxima_fill[:,:,i]>0
    #tmp_mask=morphology.binary_dilation(tmp_mask,morphology.disk(3))
    markers_r[tmp_mask]=r_list[i]