"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import cv2

def get_boundary(image):
    boundary_pts,_ = cv2.findContours(np.uint8(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_pts = np.vstack(boundary_pts)
    boundary_pts = boundary_pts.reshape(len(boundary_pts),2)
    return boundary_pts
