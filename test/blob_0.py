# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:27:39 2017

@author: Szabolcs
"""
#scale=1
#diag.gray, scale=imtools.imRescaleMaxDim(diag.im_corrected[:,:,1], diag.param.small_size,\
#                                         interpolation=0)
#diag.param.rbcR=diag.blob_detection(255-diag.gray,scale=scale,max_res=100,min_res=25,\
#                                            threshold=0.1, vis_diag=True)  
#
#
#mask_fg_clear=label_fg_bg>3
#cell_morphology.rbc_mask_morphology(im_resize,label_fg_bg,diag.param,label_tsh=3,vis_diag=vis_diag,fig='31')    
 
#max_res=150
#min_res=25
#threshold=0.001
#blobs = feature.blob_log((255*mask_fg_clear).astype('uint8'),\
#                        min_sigma=max(mask_fg_clear.shape)/max_res,\
#                        max_sigma=max(mask_fg_clear.shape)/min_res,\
#                        threshold=threshold, overlap=1)
#         #    blobs = feature.blob_dog(gray, max_sigma=20, threshold=.1)
#         #    blobs = feature.blob_doh(gray, max_sigma=30, threshold=.005)
#
## Compute radii in the 3rd column.
#blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
#
#
#hist_r,bin_edges=np.histogram(blobs[:,2]/scale,20)
#hist_r=hist_r[1:-1]
#bin_edges=bin_edges[1:-1]
#rMAX=((bin_edges[np.argmax(hist_r)]+bin_edges[np.argmax(hist_r)+1])/2)
#
#blobs_2=blobs[np.logical_and(blobs[:,2]>0.75*rMAX*scale,blobs[:,2]<3*rMAX*scale),:]
#
#
#fi=plt.figure('blobs')
#fi.clear()
#ax1=fi.add_subplot(121)
#ax1.imshow(mask_fg_clear,cmap='gray')
#for blob in blobs_2:
#    y, x, r = blob
#    c = plt.Circle((x, y), r, color='g', linewidth=2, fill=False)
#    ax1.add_patch(c)
#    ax1.set_axis_off()
#ax2=fi.add_subplot(122)
#ax2.bar(bin_edges[:-1], hist_r, width = 1)
#plt.tight_layout()
#plt.show()
    

##

mask_fg_clear=label_fg_bg>30
#mask_fg_clear=cell_morphology.rbc_mask_morphology(im,label_fg_bg_orig,diag.param,label_tsh=3,vis_diag=vis_diag,fig='31')    

min_r=int(max(mask_fg_clear.shape)/scale/100)
max_r=int(max(mask_fg_clear.shape)/scale/30)
r_list = np.linspace(min_r, max_r, (max_r-min_r)+1)

from scipy import ndimage
#k=morphology.disk(5)
#c=ndimage.convolve((mask_fg_clear).astype('float64'), k, mode='constant', cval=0.0)             
#             
image = (mask_fg_clear).astype('float64')

#if log_scale:
#    start, stop = log(min_sigma, 10), log(max_sigma, 10)
#    sigma_list = np.logspace(start, stop, num_sigma)
#else:

# computing gaussian laplace
# s**2 provides scale invariance
im_filtered = [ndimage.convolve(image, morphology.disk(r*scale))/(morphology.disk(r*scale).sum()) for r in r_list]


# computing difference between two successive Gaussian blurred images
# multiplying with standard deviation provides scale invariance
#from scipy.ndimage import gaussian_filter, gaussian_laplace
#im_filtered = [-gaussian_laplace(255-im_resize, r*scale) * (r*scale) ** 2 for r in r_list]


fill_cube = np.dstack(im_filtered)
delta_cube = fill_cube[:,:,1:]-fill_cube[:,:,0:-1]

inc_cube=np.cumsum(delta_cube>=0, axis=2)

label_inc_stop=np.zeros(mask_fg_clear.shape)
fill_at_max=np.zeros(mask_fg_clear.shape)

for i in range(delta_cube.shape[2]):
    label_inc_stop[inc_cube[:,:,i]==i]=i
    fill_at_max[inc_cube[:,:,i]==i]=fill_cube[inc_cube[:,:,i]==i,i]          


label_inc_stop[fill_at_max<0.75]=0

threshold=0
local_maxima_1 = feature.peak_local_max(label_inc_stop, threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((11,11)),
                                      threshold_rel=0.0,
                                      exclude_border=False)

threshold=0.75
local_maxima_2 = feature.peak_local_max(fill_cube, threshold_abs=threshold,
                                      indices=False,
                                      footprint=np.ones((3,3,1)),
                                      threshold_rel=0.0,
                                      exclude_border=False)


markers_1 = measure.label(local_maxima_1,return_num=False)
markers_2=measure.label(np.sum(local_maxima_2,axis=2),return_num=False)  
#markers_2 = morphology.binary_closing(markers_2,morphology.disk(3))
markers_2 = morphology.binary_opening(markers_2,morphology.disk(3))
#markers_2 = morphology.binary_dilation(markers_2,morphology.disk(3))
#markers_2 = morphology.binary_opening(markers_2,morphology.disk(1))

from skimage import color
#wbc_blob=imtools.overlayImage(color.gray2rgb(255*mask_fg_clear).astype('uint8'),markers_2>0,(0,1,0),1,vis_diag=True,fig='blob_mask')
wbc_blob=imtools.overlayImage(im_resize,markers_2>0,(0,1,0),1,vis_diag=True,fig='blob_mask')

wbc_blob=imtools.overlayImage(wbc_blob,markers_1>0,(1,0,0),1,vis_diag=True,fig='blob_mask')
   

markers=np.logical_and(markers_1>0,markers_2>0)

stops=label_inc_stop[markers>0].flatten()

hh, bin_edges=np.histogram(stops,20)
fi=plt.figure('hh')
ax=fi.add_subplot(111)
ax.plot(bin_edges[1:],hh)

from scipy import signal
i1=signal.argrelmax(hh)
i2=np.argmax(hh[i1])
i_max=i1[0][i2]

   
rMAX=r_list[np.round(bin_edges[i_max+1])]


##
#mag=imtools.getGradientMagnitude(im_resize[:,:,1])

#mag_tsh=np.median(mag)*5

markers_3=markers.copy()
#markers_2[morphology.binary_dilation(mag>mag_tsh,morphology.disk(1))==1]=0
markers_3[np.logical_or(label_inc_stop<bin_edges[i_max]-10,label_inc_stop>bin_edges[i_max]+10)]=0
#wbc_blob=imtools.overlayImage(color.gray2rgb(255*mask_fg_clear).astype('uint8'),morphology.binary_dilation(markers_2>0,morphology.disk(5)),(1,0,0),1,vis_diag=True,fig='blob_mask')
wbc_blob=imtools.overlayImage(im_resize,morphology.binary_dilation(markers_3>0,morphology.disk(3)),(1,0,0),1,vis_diag=True,fig='rbc_mask')
#wbc_blob=imtools.overlayImage(wbc_blob,morphology.binary_dilation(mag>mag_tsh,morphology.disk(1))==1,(0,1,0),1,vis_diag=True,fig='rbc_mask')

#mag=imtools.getGradientMagnitude(hsv_resize[:,:,1])

#
#fill_2d=fill_cube[:,:,np.round(bin_edges[i_max+1])]
#wbc_blob=imtools.overlayImage(im_resize,fill_2d>0.9,(1,0,0),1,vis_diag=True,fig='blob_mask')

###########
#mark_stop=markers>0
#mark_stop[label_inc_stop<i_max-1]=0
#
#        
#(mask_fg_clear>0).sum()/mask_fg_clear.size
#mask3d=np.zeros(fill_cube.shape)
#mask3d[fill_cube>np.pi*0.65]=1
#    
#mask2d=np.sum(mask3d,axis=(0,1))    
#plt.plot(r_list,mask2d)
#
#threshold=3
#
#local_maxima = feature.peak_local_max(image_cube, threshold_abs=threshold,
#                              footprint=np.ones((9,9, 5)),
#                              threshold_rel=0.0,
#                              exclude_border=False)
#
#lm = local_maxima.astype(np.float64)
## Convert the last index to its corresponding scale value
#lm[:, 2] = r_list[local_maxima[:, 2]]
#blobs = lm
#
#hist_r,bin_edges=np.histogram(blobs[:,2]/scale,10)
#hist_r=hist_r[1:-1]
#bin_edges=bin_edges[1:-1]
#rMAX=((bin_edges[np.argmax(hist_r)]+bin_edges[np.argmax(hist_r)+1])/2)
#
#
#blobs_2=blobs[np.logical_and(blobs[:,2]>2,blobs[:,2]<30),:]
#
#
#fi=plt.figure('blobs')
#fi.clear()
#ax1=fi.add_subplot(121)
#ax1.imshow(mask_fg_clear,cmap='gray')
#for blob in blobs:
#    y, x, r = blob
#    c = plt.Circle((x, y), r, color='g', linewidth=2, fill=False)
#    ax1.add_patch(c)
#    ax1.set_axis_off()
#ax2=fi.add_subplot(122)
#ax2.bar(bin_edges[:-1], hist_r, width = 1)
#plt.tight_layout()
#plt.show()