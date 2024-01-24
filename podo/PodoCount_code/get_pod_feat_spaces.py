"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np

def get_pod_feat_spaces(total_gloms):

    pod_feat_labels = ['raw_pod_cts','pods_per_glom_area','total_pod_area','pod_glom_area_ratio','pod_areas','pod_bb_areas','pod_convex_areas','pod_eccentricities','pod_equiv_diams','pod_extents','pod_major_axis_lengths','pod_minor_axis_lengths','pod_max_intensities','pod_mean_intensities','pod_min_intensities','pod_orientations','pod_perimeters','pod_solidities']
    pod_feat_qty = len(pod_feat_labels)
    pod_feat_array = np.empty([pod_feat_qty,total_gloms])

    return pod_feat_labels, pod_feat_qty, pod_feat_array