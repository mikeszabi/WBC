# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:56:35 2017

@author: SzMike
"""

from scipy.ndimage.filters import generic_filter
import functools


sum_filter = functools.partial(generic_filter,
                                  function=np.median,
                                  footprint=fp)

generic_filter(mask_wbc_nucleus,
                       function=sum,
                       footprint=fp,
                       mode='nearest')

from skimage.filters import rank
from skimage import img_as_float



circ=morphology.disk(30)
circ_response = 255*img_as_float(rank.mean(mask_wbc_nucleus>0, selem=circ, mask=(label_mask>1)))
circ_response.flat[(circ_response>0).flatten()]+=np.random.random(((circ_response>0).sum()))
local_maxi = feature.peak_local_max(circ_response, indices=False, 
                                        threshold_abs=30,
                                        footprint=np.ones((int(2*param.rbcR), int(2*param.rbcR))), 
                                        labels=mask_wbc_nucleus.copy())

markers_WBC, n_WBC = measure.label(local_maxi,return_num=True)


im_detect=imtools.overlayImage(im,morphology.binary_dilation(markers_WBC>0,morphology.disk(5)),\
            (1,0,0),1,vis_diag=vis_diag,fig='detections')