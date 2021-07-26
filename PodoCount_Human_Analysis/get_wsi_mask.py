"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
import scipy as sp
import cv2
from skimage import color
from skimage.transform import resize

def get_wsi_mask(image,wsi_glom_mask):
    wsi_mask = color.rgb2gray(image)
    wsi_mask = wsi_mask<(wsi_mask.mean())
    wsi_mask = sp.ndimage.morphology.binary_fill_holes(wsi_mask)
    wsi_mask = (resize(wsi_mask,wsi_glom_mask.shape,anti_aliasing=True)>0.01)*1
    wsi_mask = np.logical_and(wsi_mask,(1-wsi_glom_mask))
    wsi_labels, num_labels = sp.ndimage.label(wsi_mask)
    wsi_props = sk.measure.regionprops(wsi_labels)
    euler_labels = np.empty([num_labels,2])
    for label in range(num_labels):
        euler_labels[label,0] = wsi_props[label].euler_number
        euler_labels[label,1] = (label+1)
    keep = np.where(euler_labels<0)[0]
    keep = keep.reshape(len(keep),1)
    wsi_mask = np.zeros(wsi_labels.shape)
    for val in range(len(keep)):
        label = keep[val,0] + 1
        wsi_mask[wsi_labels==label] = 1
    wsi_mask = sp.ndimage.morphology.binary_fill_holes(wsi_mask)
    return(wsi_mask)
