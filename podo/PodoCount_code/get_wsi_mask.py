"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
import scipy as sp
from skimage import color
from PIL import Image
from skimage.transform import resize

def get_wsi_mask(image,wsi_glom_mask):
    wsi_mask = color.rgb2gray(image)
    wsi_mask = wsi_mask<(wsi_mask.mean())
    wsi_mask = sp.ndimage.morphology.binary_fill_holes(wsi_mask)
    #wsi_mask = (resize(wsi_mask,wsi_glom_mask.shape,anti_aliasing=False)>0.01)*1
    
    # Convert the mask to uint8 format, scaling values to 0-255
    wsi_mask_uint8 = (wsi_mask * 255).astype(np.uint8)

    # Create a PIL image for resizing
    pil_image = Image.fromarray(wsi_mask_uint8)

    # Dimensions for resizing should match the target mask's shape
    width, height = wsi_glom_mask.shape

    # Resizing the mask using PIL's resize method with LANCZOS resampling
    resized_wsi_mask = pil_image.resize((height, width), Image.Resampling.LANCZOS)

    # Convert back to numpy array
    resized_wsi_mask = np.array(resized_wsi_mask)

    # Apply a threshold to get back a binary mask
    # The threshold is 0.01 * 255 to match the scale of the original mask
    threshold = 0.01 * 255
    wsi_mask = (resized_wsi_mask > threshold) * 1
    wsi_mask = np.logical_and(wsi_mask,(1-wsi_glom_mask)) #change till here. Lines 16-22
    
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
