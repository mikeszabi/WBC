# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:20:02 2017

@author: SzMike
"""

import os

class prp:
    def __init__(self):
        self.pixelSize=1 # in microns
        self.magnification=1
        self.rbcR=25
        self.wbcRatio=0.8
        self.project='WBC'
        self.root_dir=r'd:\Projects'
    
    def getTestImageDirs(self,wbc_type='Lymphocyte'):
        data_dir=os.path.join(self.root_dir,self.project,'data')
        image_dir=os.path.join(data_dir,'Test','WBC Types',wbc_type)
        return image_dir