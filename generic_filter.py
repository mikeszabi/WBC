# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:56:35 2017

@author: SzMike
"""

from scipy.ndimage.filters import generic_filter
import functools

fp=morphology.disk(30)

sum_filter = functools.partial(generic_filter,
                                  function=np.median,
                                  footprint=fp)

generic_filter(mask_wbc_nucleus,
                       function=sum,
                       footprint=fp,
                       mode='nearest')