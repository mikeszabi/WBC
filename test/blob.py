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
mask_fg_clear=label_fg_bg>3
#cell_morphology.rbc_mask_morphology(im_resize,label_fg_bg,diag.param,label_tsh=3,vis_diag=vis_diag,fig='31')    
 
max_res=150
min_res=25
threshold=0.001
blobs = feature.blob_log((255*mask_fg_clear).astype('uint8'),\
                        min_sigma=max(mask_fg_clear.shape)/max_res,\
                        max_sigma=max(mask_fg_clear.shape)/min_res,\
                        threshold=threshold, overlap=1)
         #    blobs = feature.blob_dog(gray, max_sigma=20, threshold=.1)
         #    blobs = feature.blob_doh(gray, max_sigma=30, threshold=.005)

# Compute radii in the 3rd column.
blobs[:, 2] = blobs[:, 2] * math.sqrt(2)


hist_r,bin_edges=np.histogram(blobs[:,2]/scale,20)
hist_r=hist_r[1:-1]
bin_edges=bin_edges[1:-1]
rMAX=((bin_edges[np.argmax(hist_r)]+bin_edges[np.argmax(hist_r)+1])/2)

blobs_2=blobs[np.logical_and(blobs[:,2]>0.75*rMAX*scale,blobs[:,2]<3*rMAX*scale),:]


fi=plt.figure('blobs')
fi.clear()
ax1=fi.add_subplot(121)
ax1.imshow(mask_fg_clear,cmap='gray')
for blob in blobs_2:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='g', linewidth=2, fill=False)
    ax1.add_patch(c)
    ax1.set_axis_off()
ax2=fi.add_subplot(122)
ax2.bar(bin_edges[:-1], hist_r, width = 1)
plt.tight_layout()
plt.show()
    