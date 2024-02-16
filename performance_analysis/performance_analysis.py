import numpy as np
import openslide
import pandas as pd
import cv2
from skimage.measure import label,regionprops
from skimage.measure import label,regionprops
from getMaskFromXml import getMaskFromXml


sourcePAS = openslide.open_slide('2__c20_27507_1_____ - 2020-11-25 19.56.09.ndpi')
sourcePAS_shifted = openslide.open_slide('2__2020-12-17 20.04.00 2021-01-13 15.29.12.ndpi')

mask = getMaskFromXml(sourcePAS,'original.xml')  
mask_glom = getMaskFromXml(sourcePAS,'original_glom.xml')   

mask_shifted = getMaskFromXml(sourcePAS_shifted,'shifted.xml')  
mask_glom_shifted = getMaskFromXml(sourcePAS_shifted,'shifted_glom.xml') 

nuclei = getMaskFromXml(sourcePAS,'nuclei.xml') 

all_regions= []
all_gloms = []
for reg in regionprops(label(mask_glom)): 
    minr, minc, maxr, maxc = reg.bbox
    ptx = (minr+maxr)/2
    pty = (minc+maxc)/2
    highres_w = 800
    rs = mask[int(ptx-(highres_w/2)):int(ptx+(highres_w/2)),int(pty-(highres_w/2)):int(pty+(highres_w/2))] 
    all_regions.append(mask_glom[int(ptx-(highres_w/2)):int(ptx+(highres_w/2)),int(pty-(highres_w/2)):int(pty+(highres_w/2))]*rs)
    all_gloms.append(mask_glom[int(ptx-(highres_w/2)):int(ptx+(highres_w/2)),int(pty-(highres_w/2)):int(pty+(highres_w/2))])

all_regions_shifted= []
for reg in regionprops(label(mask_glom_shifted)): 
    minr, minc, maxr, maxc = reg.bbox
    ptx = (minr+maxr)/2
    pty = (minc+maxc)/2
    highres_w = 800
    rs = mask_shifted[int(ptx-(highres_w/2)):int(ptx+(highres_w/2)),int(pty-(highres_w/2)):int(pty+(highres_w/2))] 
    all_regions_shifted.append(mask_glom_shifted[int(ptx-(highres_w/2)):int(ptx+(highres_w/2)),int(pty-(highres_w/2)):int(pty+(highres_w/2))]*rs)

all_nuclei= []
for reg in regionprops(label(mask_glom)): 
    minr, minc, maxr, maxc = reg.bbox
    ptx = (minr+maxr)/2
    pty = (minc+maxc)/2
    highres_w = 800
    rs = nuclei[int(ptx-(highres_w/2)):int(ptx+(highres_w/2)),int(pty-(highres_w/2)):int(pty+(highres_w/2))] 
    all_nuclei.append(mask_glom[int(ptx-(highres_w/2)):int(ptx+(highres_w/2)),int(pty-(highres_w/2)):int(pty+(highres_w/2))]*rs)
    
all_tp=[]
all_fp=[]
all_fn=[]
all_tn=[]
all_sensitivity=[]
all_specifity=[]
all_gloms=[]
glom_coord=[]

for k in range(len(regionprops(label(mask_glom)))):

    base = np.array(np.logical_or(all_nuclei[k],all_regions_shifted[k])*255, dtype='uint8')
    ret, thresh = cv2.threshold(base, 128, 255, 0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for i in range(len(contours)):
        cnt = contours[i]
        tmp = np.zeros_like(base)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        ret, thresh = cv.threshold(base, 128, 255, 0)
        contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv.drawContours(tmp, [cnt], -1, (255,255,255), thickness=cv2.FILLED)
        tmp = cv2.cvtColor(np.array(tmp/255, dtype='uint8'), cv2.COLOR_BGR2GRAY)
        check_pod_pred = tmp*(all_regions[k]*1)
        check_pod = tmp*(all_regions_shifted[k]*1)
        pod = np.sum(check_pod[:])>15
        pod_pred = np.sum(check_pod_pred[:])>15

        if pod & pod_pred:
            tp+=1
        elif pod:
            fn+=1
        elif pod_pred:
            fp+=1
        else:
            tn+=1
    all_tp.append(tp)
    all_fp.append(fp)
    all_fn.append(fn)
    all_tn.append(tn)
    all_sensitivity.append(round(tp/(tp+fn+0.01),2))
    all_specifity.append(round(tn/(tn+fp+0.01),2))
    all_gloms.append('glom_'+str(k))
    glom_coord.append(regionprops(label(mask_glom))[k].bbox)


dataframe = pd.DataFrame()
dataframe['gloms']=all_gloms
dataframe['true_positives']=all_tp
dataframe['false_positives']=all_fp
dataframe['false_negatives']=all_fn
dataframe['true_negatives']=all_tn
dataframe['sensitivity']=all_sensitivity
dataframe['specifity']=all_specifity
dataframe['glom_bbox'] = glom_coord
    