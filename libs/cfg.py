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
        self.cellFillAreaPct=0.25
        self.cellOpeningPct=0.33
        self.small_size=128
        self.middle_size=256
        self.rgb_range_in_hue=((-30/360,30/360), (75/360,135/360), (180/360,240/360))
        self.wbc_range_in_hue=(225/360,330/360)
        self.hueWidth=2
        self.project='WBC'
        #self.root_dir=r'd:\Projects'
        self.root_dir=os.curdir
        self.data_dir=r'd:\DATA'
    
    def getTestImageDirs(self,wbc_type=''):
        test_dir=os.path.join(self.root_dir,'data')
        image_dir=os.path.join(test_dir,'Test','WBC Types',wbc_type)
        return image_dir
    
    def getImageDirs(self,dir_name=''):
        image_dir=os.path.join(self.data_dir,dir_name)
        return image_dir
    
    def getOutDir(self,dir_name=''):
        save_dir=os.path.join(self.root_dir,dir_name)
        return save_dir