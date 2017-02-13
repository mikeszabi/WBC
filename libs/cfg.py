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
    
    def getSaveDir(self,dir_name=''):
        save_dir=os.path.join(self.root_dir,'diag')
        return save_dir