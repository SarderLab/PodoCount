"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
from numpy import linalg
from skimage.util import dtype
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
import cv2

pasdab_rgb = np.array([[0.3697298, 0.61498046, 0.69649047],
                         [0.4091243, 0.8440652, 0.34665722],
                         [0.47995895, 0.6926196, 0.5384398]])

pasdab_rgb = linalg.inv(pasdab_rgb)

#Functions
def separate_stains(rgb, color_deconv_vector):
    rgb = dtype.img_as_float(rgb, force_copy=True)
    rgb += 2
    stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), color_deconv_vector)
    return np.reshape(stains, rgb.shape)

def stain_norm(ihc_3, channel):
    rescale = rescale_intensity(ihc_3[:, :, channel], out_range=(0,1))
    stain_array = np.dstack((np.zeros_like(rescale), rescale, rescale))
    grey_array = rgb2gray(stain_array)
    return grey_array

def stain_decon(ihc):
    ihc_decon = separate_stains(ihc, pasdab_rgb)
    res = stain_norm(ihc_decon, 0)
    pas = stain_norm(ihc_decon, 1)
    ihc = stain_norm(ihc_decon, 2)

    return res,pas,ihc
