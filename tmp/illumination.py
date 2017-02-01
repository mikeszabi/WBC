# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:33:32 2017

@author: SzMike
"""

def doMorphology(mask):
    r=int(max(mask.shape)/10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def overMask(intensity_image):
    img_tmp=np.empty(intensity_image.shape, dtype='uint8')   
    mask_o=np.empty(intensity_image.shape, dtype='bool') 
    mask_o=intensity_image==255
    mask_o=255*mask_o.astype(dtype=np.uint8)
    #mask_o=doMorphology(mask_o)
  
    return mask_o

# rescale image
plotFlag=True

im_s, scale = tools.imresizeMaxDim(im, 256)

rgb = cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB)

if plotFlag:
    fo=plt.figure('rgb')
    axo=fo.add_subplot(111)
    axo.imshow(rgb)

im_cs = cv2.cvtColor(im_s, cv2.COLOR_BGR2HSV)

Z = im_cs.reshape((-1,3))


hist = tools.colorHist(im_cs,plotFlag=True,mask=255-mask_o)

# overexpo mask
mask_o=overMask(im_cs[:,:,2])
tools.maskOverlay(rgb,mask_o,0.5,1,sbs=False,plotFlag=True,ax='none')

# KMEANS on saturation and intensity
Z = im_cs.reshape((-1,3))
Z = np.float32(Z)/256
Z_mask=mask_o.reshape((-1,1))==0
Z_mask=Z_mask.flatten()

# select channels
Z_1=Z[Z_mask,1:3]
Z=Z[:,1:3]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z_1,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
# TODO: initialize centers from histogram peaks
center = np.uint8(center*256)
print(center)
#res = center[label.flatten()]
#res2 = res.reshape((im.shape))

lab_all=np.zeros(Z.shape[0])
lab_all.flat[Z_mask==False]=-1
lab_all.flat[Z_mask==True]=label
       
# not overexposed mask
maxi=np.argmax(center[:,1])
sure_bg_mask = lab_all.reshape((im_cs.shape[0:2]))==maxi
sure_bg_mask = tools.normalize(sure_bg_mask.astype('uint8'),1)
tools.maskOverlay(rgb,sure_bg_mask,0.5,1,sbs=False,plotFlag=True,ax='none')

maxi=np.argmax(center[:,0])
sure_fg_mask = lab_all.reshape((im_cs.shape[0:2]))==maxi
sure_fg_mask = tools.normalize(sure_fg_mask.astype('uint8'),1)
tools.maskOverlay(rgb,sure_fg_mask,0.5,1,sbs=False,plotFlag=True,ax='none')

hist = tools.colorHist(im_cs[:,:,2],plotFlag=True,mask=sure_bg_mask)
center[maxi]

illumination_inhomogenity=im_cs[:,:,2].astype('int')-center[maxi,1].astype('int')
illumination_inhomogenity[sure_bg_mask==False]=0
illumination_inhomogenity[mask_o==255]=255-center[maxi,1].astype('int')
tools.normalize(illumination_inhomogenity,1)

