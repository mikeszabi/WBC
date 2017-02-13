# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:57:09 2017

@author: SzMike
"""

import warnings
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.exposure import cumulative_distribution
from skimage import filters, img_as_ubyte
from mpl_toolkits.axes_grid1 import make_axes_locatable

# colorhist  - works for grayscale and color images
def colorHist(im,vis_diag=False,mask=None,fig=''):
    assert im.ndim==3, 'Not 3channel image'
    color = ('r','g','b')
    histr=[]
    if len(im.shape)==2:
        nCh=1
    else:
        nCh=3    
    for i in range(nCh):
        if mask is None:
            im_masked=im[:,:,i]
        else:
            im_masked=im[:,:,i].flat[mask.flatten()>0]
        h, b=np.histogram(im_masked,bins=range(255),density=True)
        histr.append(h)
        if vis_diag:
            fh=plt.figure(fig+'_histogram')
            ax=fh.add_subplot(111)
            ax.plot(histr[i],color = color[i])
            ax.set_xlim([0,256])
    return histr
    
def maskOverlay(im,mask,alpha,ch=1,sbs=False,ax=None, vis_diag=False,fig=''):
# mask is 2D binary
# image can be 1 or 3 channel
# ch : rgb -> 012
# sbs: side by side
# http://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
    if ch>2:
        ch=1

    mask_tmp=np.empty(mask.shape+(3,), dtype='uint8')   
    mask_tmp.fill(0)
    mask_tmp[:,:,ch]=mask
    
    if im.ndim==2:
        im_3=np.matlib.repeat(np.expand_dims(im,2),3,2)
    else:
        im_3=im
        
    im_overlay=np.add(alpha*mask_tmp,(1-alpha)*im_3).astype(im.dtype)
    
    if vis_diag:
        if sbs:
            both = np.hstack((im_3,im_overlay))
        else:
            both=im_overlay
        if ax is None:
            fi=plt.figure(fig+'_overlayed')
            ax=fi.add_subplot(111)
        ax.imshow(both)
    return im_overlay
    
def normalize(im,vis_diag=False,ax=None,fig=''):
    # normalize intensity image
    assert im.ndim==2, 'Not 1channel image'
    cdf, bins=cumulative_distribution(im, nbins=256)
    minI=bins[np.argwhere(cdf>0.01)[0,0]]
    maxI=bins[np.argwhere(cdf>0.99)[0,0]]
    im_norm=im.copy()
    im_norm[im_norm<minI]=minI
    im_norm[im_norm>maxI]=maxI
    im_norm=(im_norm-minI)/(maxI-minI)
    im_norm=(255*im_norm).astype('uint8')       
    if vis_diag:
        if ax is None:
            fi=plt.figure(fig+'_normalized')
            ax=fi.add_subplot(111)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            i=ax.imshow(im_norm,cmap='jet')
            fi.colorbar(i, cax=cax, orientation='vertical')
        else:
            ax.imshow(im_norm,cmap='jet')
        plt.show()
    return im_norm            
    
def getGradientMagnitude(im):
    #Get magnitude of gradient for given image"
    assert len(im.shape)==2, "Not 2D image"
    mag = filters.scharr(im)
    return mag


def imRescaleMaxDim(im, maxDim, boUpscale = False, interpolation = 1):
    scale = 1.0 * maxDim / max(im.shape[:2])
    if scale < 1  or boUpscale:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = img_as_ubyte(rescale(im, scale, order=interpolation))
    else:
        scale = 1.0
    return im, scale

def plotShapes(im, shapelist):
    fs=plt.figure('shapes image')
    axs=fs.add_subplot(111)
    axs.imshow(im)  
    for shape in shapelist:
        pts=shape[2]
        axs.plot(pts[:,1], pts[:, 0], linewidth=3, color='g')

def histogram_similarity(hist, reference_hist):
   
    # Compute Chi squared distance metric: sum((X-Y)^2 / (X+Y));
    # a measure of distance between histograms
    X = hist
    Y = reference_hist

    num = (X - Y) ** 2
    denom = X + Y
    denom[denom == 0] = np.infty
    frac = num / denom

    chi_sqr = 0.5 * np.sum(frac, axis=0)

    # Generate a similarity measure. It needs to be low when distance is high
    # and high when distance is low; taking the reciprocal will do this.
    # Chi squared will always be >= 0, add small value to prevent divide by 0.
    similarity = 1 / (chi_sqr + 1.0e-4)

    return similarity
