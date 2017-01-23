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
        self.wbcRatio=0.8
        self.project='WBC'
        self.root_dir=r'd:\Projects'
        self.data_dir=r'd:\DATA'
    
    def getTestImageDirs(self,wbc_type=''):
        test_dir=os.path.join(self.root_dir,self.project,'data')
        image_dir=os.path.join(test_dir,'Test','WBC Types',wbc_type)
        return image_dir
    
    def getImageDirs(self,dir_name=''):
        image_dir=os.path.join(self.data_dir,dir_name)
        return image_dir