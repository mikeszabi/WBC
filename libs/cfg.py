# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:20:02 2017

@author: SzMike
"""

import os

class param:
    def __init__(self):
        self.pixelSize=1 # in microns
        self.magnification=1
        self.rbcR=25
        self.cellFillAreaPct=0.5
        self.cellOpeningPct=0.33
        self.small_size=128
        self.middle_size=512
        self.cell_bound_pct=0.25
        self.over_saturated_rbc_ratio=0.75
        self.rgb_range_in_hue=((-30/360,30/360), (75/360,135/360), (180/360,240/360))
        self.wbc_range_in_hue=(230/360,260/360)
        self.hueWidth=2
        self.project='WBC'
        self.root_dir=os.curdir
        self.data_dir=os.path.join(self.root_dir,'data','Test','WBC Types')
    
    def getImageDirs(self,data_dir=None, dir_name=''):
        if data_dir is None:
            data_dir=self.data_dir
        image_dir=os.path.join(data_dir,dir_name)
        return image_dir
    
    def getOutDir(self,data_dir=None,dir_name=''):
        if data_dir is None:
            data_dir=self.root_dir
        save_dir=os.path.join(self.root_dir,dir_name)
        if not os.path.exists(save_dir):
            print('directory is created')
            os.makedirs(save_dir)
        return save_dir