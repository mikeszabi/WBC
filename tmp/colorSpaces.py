# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:01:22 2017

@author: SzMike
"""

from skimage import color
ihc_hed = color.rgb2hed(im)
ihc_rgb = color.hed2rgb(ihc_hed)

img_xyz = color.convert_colorspace(im, 'RGB', 'XYZ')
img_lab = color.convert_colorspace(img_xyz, 'XYZ', 'YUV')

color.colorconv.lab_ref_white = np.array([0.96422, 1.0, 0.82521])
lab = color.rgb2yiq(img_as_float(im))