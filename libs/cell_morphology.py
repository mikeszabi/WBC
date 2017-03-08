# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:48:17 2017

@author: SzMike
"""

from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
from skimage import color
from scipy import ndimage
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


import imtools
import segmentations


def rbc_labels(im,clust_centers_0,label_0,vis_diag=False):
    # creating meaningful labels for foreground-background segmentation and RBC detection
    cent_dist=segmentations.center_diff_matrix(clust_centers_0,metric='euclidean')
    
    # adding meaningful labels
    ind_sat=np.argsort(clust_centers_0[:,0])
    ind_val=np.argsort(clust_centers_0[:,2])
    
    label_fg_bg=np.zeros(label_0.shape).astype('uint8')
    label_fg_bg[label_0==ind_val[-2]]=2 # unsure region
    label_fg_bg[label_0==ind_sat[-3]]=2 # unsure region
    label_fg_bg[label_0==ind_val[-1]]=1 # sure background
    label_fg_bg[label_0==ind_sat[-1]]=31 # cell foreground guess 1 
    if cent_dist[ind_sat[-1],ind_sat[-2]]<cent_dist[ind_sat[-2],ind_val[-1]]:
       label_fg_bg[label_0==ind_sat[-2]]=32 # cell foreground guess 2
       if cent_dist[ind_sat[-2],ind_sat[-3]]<cent_dist[ind_sat[-3],ind_val[-1]]:                 
           label_fg_bg[label_0==ind_sat[-3]]=33 # cell foreground guess 3
          
    return label_fg_bg

def wbc_markers(mask_fg,param,fill_tsh=0.25,scale=1,vis_diag=False,fig=''):

    mask_fg=morphology.binary_opening(mask_fg,morphology.disk(2))
    mask_fg=morphology.binary_closing(mask_fg,morphology.disk(2))

    max_dim=max(mask_fg.shape)
    
    min_r=int(max(mask_fg.shape)/scale/50)
    max_r=int(max(mask_fg.shape)/scale/20)
    r_list = np.linspace(min_r, max_r, (max_r-min_r)+1)
       
    start_r=0
    r_list=r_list[r_list>start_r]
          
    mask = (mask_fg).astype('float64')
    
    im_filtered = [ndimage.convolve(mask, morphology.disk(r*scale))/(morphology.disk(r*scale).sum()) for r in r_list]
    
    fill_cube = np.dstack(im_filtered)
    
    fp=int(max_dim/scale/50)   
    threshold=fill_tsh
    local_maxima_fill = feature.peak_local_max(fill_cube, 
                                          threshold_abs=threshold,
                                          indices=True,
                                          footprint=np.ones((fp,fp,3)),
                                          threshold_rel=0.0,
                                          exclude_border=False)
    
    if vis_diag:
        fig=plt.figure('circle image')
        axs=fig.add_subplot(111)
        axs.imshow(color.gray2rgb(255*mask).astype('uint8'))  
        for l in local_maxima_fill:
            circ=plt.Circle((l[1],l[0]), radius=r_list[l[2]]*scale, color='g', fill=True)
            axs.add_patch(circ)
            
    markers_r=np.zeros(mask_fg.shape)    
    for l in local_maxima_fill:
        markers_r[l[0],l[1]]=r_list[l[2]]    
    markers=morphology.binary_dilation(markers_r,morphology.disk(3)).astype('uint8')
    markers[markers_r>0]=markers_r[markers_r>0]
        
    return markers
        
def blob_markers(mask_fg,param,rbc=True,fill_tsh=0.75,scale=1,vis_diag=False,fig=''):
    
#    mask_fg=morphology.binary_closing(mask_fg,morphology.disk(2))
#    mask_fg=morphology.binary_opening(mask_fg,morphology.disk(2))    
#    
    max_dim=max(mask_fg.shape)
    
    min_r=int(max_dim/scale/100)
    max_r=int(max_dim/scale/30)
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
    fp=int(max_dim/scale/150)
    local_maxima_inc = feature.peak_local_max(inc_cube, threshold_abs=threshold,
                                          indices=False,
                                          footprint=np.ones((fp,fp,3)),
                                          threshold_rel=0.0,
                                          exclude_border=False)
    
    threshold=fill_tsh
    fp=int(max_dim/scale/150)   
    local_maxima_fill = feature.peak_local_max(fill_cube[:,:,1:], threshold_abs=threshold,
                                          indices=False,
                                          footprint=np.ones((fp,fp,3)),
                                          threshold_rel=0.0,
                                          exclude_border=False)
    
    markers_r=np.zeros(mask_fg.shape)
    for i in range(inc_cube.shape[2]):
        tmp_mask=np.logical_and(local_maxima_inc[:,:,i]>0,local_maxima_fill[:,:,i]>0)
        #tmp_mask=morphology.binary_dilation(tmp_mask,morphology.disk(3))
        markers_r[tmp_mask]=r_list[i+1]
    markers_r_raw=markers_r.copy()        
    if rbc:
    
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
        markers_r[np.logical_or(markers_r<rbcR*0.5,markers_r>rbcR*1.5)]=0
        
        # TODO create marker list with corresponding radius
    else:
        
        #markers[label_inc_stop<np.argmax(r_list>param.rbcR*0.75)]=0
        rbcR=None
        # TODO create marker list with corresponding radius
    markers=markers_r>0  
    markers=morphology.binary_dilation(markers_r,morphology.disk(int(rbcR*scale/3))).astype('uint8')
    markers=morphology.binary_opening(markers,morphology.disk(1)).astype('uint8')

    markers[markers_r>0]=markers_r[markers_r>0]

    if vis_diag:
        if rbc:
            fi=plt.figure(fig+'histogram of max radii')
            ax=fi.add_subplot(111)
            ax.plot(r_list[np.round(bin_edges[1:]).astype('int64')],stop_hist)
            # TODO check sharpness of histogram maxima - add diagnostics problem when wide
        blob=imtools.overlayImage(color.gray2rgb(255*mask).astype('uint8'),markers_r_raw>0,(0,1,0),1,vis_diag=False,fig=fig+'blob_mask')            
        blob=imtools.overlayImage(blob,markers>0,(1,0,0),1,vis_diag=True,fig=fig+'blob_mask')
        
    # TODO: create some more statistics
    return markers, rbcR

def rbc_mask_morphology(im,label_mask,param,label_tsh=3,scale=1,vis_diag=False,fig=''):
    
    mask_fg=label_mask>label_tsh
    mask_fg_open=morphology.binary_opening(mask_fg,morphology.star(2))
#   
#    mask_fg=label_mask==32
#    mask_fg_open_2=morphology.binary_closing(mask_fg,morphology.disk(1)).astype('uint8')
#    mask_fg=np.logical_or(mask_fg_open_1,mask_fg_open_2)
#   
    mask_fg_filled=morphology.remove_small_holes(mask_fg_open>0, 
                                                 min_size=scale*param.cellFillAreaPct*param.rbcR*param.rbcR*np.pi, 
                                                 connectivity=2)
    mask_fg_clear=morphology.binary_opening(mask_fg_filled,morphology.disk(scale*param.rbcR*param.cellOpeningPct)).astype('uint8')

    if vis_diag:
        f=plt.figure(fig+'_cell_overlayed')

        ax0=f.add_subplot(221)
        imtools.normalize(label_mask,ax=ax0,vis_diag=vis_diag)
        ax0.set_title('label')

        ax1=f.add_subplot(222)
        imtools.maskOverlay(im,255*(mask_fg_open>0),0.5,ax=ax1,vis_diag=vis_diag)
        ax1.set_title('foreground') 

        ax2=f.add_subplot(223)
        imtools.maskOverlay(im,255*(mask_fg_filled>0),0.5,ax=ax2,vis_diag=vis_diag)
        ax2.set_title('filled')
                       
        ax3=f.add_subplot(224)
        imtools.maskOverlay(im,255*(mask_fg_clear>0),0.5,ax=ax3,vis_diag=vis_diag)
        ax3.set_title('clear')
        
    return mask_fg_clear
   
def rbc_markers_from_mask(mask_fg_clear,param,scale=1):

    # use dtf to find markers for watershed 
    skel, dtf = morphology.medial_axis(mask_fg_clear, return_distance=True)
    dtf.flat[(mask_fg_clear>0).flatten()]+=np.random.random(((mask_fg_clear>0).sum()))
    # watershed seeds
    # TODO - add parameters to cfg
    local_maxi = feature.peak_local_max(dtf, indices=False, 
                                        threshold_abs=0.5*param.rbcR*scale,
                                        footprint=np.ones((int(1.5*param.rbcR*scale), int(1.5*param.rbcR*scale))), 
                                        labels=mask_fg_clear.copy())
    markers, n_RBC = measure.label(local_maxi,return_num=True)
    markers=morphology.binary_dilation(markers>0,morphology.disk(3))
    
    return markers

#def wbc_masks(label_1, clust_sat,param,scale,vis_diag=False):
#    # creating masks for labels        
#    n=np.zeros(clust_sat.shape[0])
#    mask_tmps=[]
#    for i, c in enumerate(clust_sat):
#        mask_tmp=morphology.binary_opening(label_1==i,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR)))
#        mask_tmps.append(mask_tmp)
#        n[i]=mask_tmp.sum()
##        
#    ind_n=np.argsort(n)
#    ind_sat=np.argsort(clust_sat)
#   
#    if n[ind_n[-1]]/sum(n)>param.over_saturated_rbc_ratio:
#        if ind_sat[-1]==ind_n[-1]:
## normally pixels with highest saturations are candidates for wbc segments, but here rbc-s have higher sat
## TODO: add diagnostics
#            mask_pot_wbc_1=mask_tmps[ind_sat[-2]]
#            mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-3]],mask_tmps[ind_sat[-2]])
#            print('undersaturated wbc') # wbc is not at highest saturation
#        elif ind_sat[-2]==ind_n[-1]:
#            mask_pot_wbc_1=mask_tmps[ind_sat[-1]]
#            mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-3]],mask_tmps[ind_sat[-1]])
#        else:
#            mask_pot_wbc_1=mask_tmps[ind_sat[-1]]
#            mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-2]],mask_tmps[ind_sat[-1]]) 
#    else:
#        mask_pot_wbc_1=mask_tmps[ind_sat[-1]]
#        mask_pot_wbc_2=np.logical_or(mask_tmps[ind_sat[-2]],mask_tmps[ind_sat[-1]]) 
#        
#    mask_wbc_pot=[]    
## TODO: dd parameters
#    mask_wbc_pot.append(morphology.binary_opening(mask_pot_wbc_1,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR))))
#    mask_wbc_pot.append(morphology.binary_opening(mask_pot_wbc_2,morphology.disk(int(scale*param.cell_bound_pct*param.rbcR))))
#
#    return mask_wbc_pot