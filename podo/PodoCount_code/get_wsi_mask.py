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
from PIL import Image
from skimage.morphology import binary_dilation, disk

def get_wsi_mask(image,wsi_glom_mask):
    wsi_mask = color.rgb2gray(image)
    wsi_mask = wsi_mask<(wsi_mask.mean())
    wsi_mask_before = wsi_mask
    wsi_mask = sp.ndimage.morphology.binary_fill_holes(wsi_mask)
    wsi_mask_after = wsi_mask
    wsi_mask_uint8 = (wsi_mask * 255).astype(np.uint8)
    pil_image = Image.fromarray(wsi_mask_uint8)
    print(pil_image.size, "before")
    width, height = wsi_glom_mask.shape
    wsi_mask = pil_image.resize((height, width), Image.Resampling.LANCZOS)
    wsi_mask = np.array(wsi_mask)
    wsi_mask = wsi_mask // 255
    print(wsi_mask.shape, 'after')
    print(wsi_glom_mask.shape)
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