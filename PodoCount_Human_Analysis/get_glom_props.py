"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
import scipy as sp
import cv2

###GLOM FEATURE ENGINEERING AND EXTRACTION###
def get_glom_props(glom_image,tissue_image,num_sections,dist_mpp,area_mpp2,df2):
    if num_sections >1:

        se = sk.morphology.disk(3)
        pod_im = sk.morphology.binary_dilation(glom_image,selem=se,out=None)

        sections_label, section_count= sp.ndimage.label(tissue_image)
        s1_gloms = np.logical_and(sections_label==1,glom_image)
        s2_gloms = np.logical_and(sections_label==2,glom_image)

        s1_glom_labels,s1_glom_count = sp.ndimage.label(s1_gloms)
        s2_glom_labels,s2_glom_count = sp.ndimage.label(s2_gloms)
        total_gloms = s1_glom_count + s2_glom_count

        glom_feat_array = []
        s_label = []
        glomID = []
        areas = []
        bbs = []
        bb_areas = []
        centroids = []
        convex_areas = []
        eccentricities = []
        equiv_diams = []
        extents = []
        major_axis_lengths = []
        minor_axis_lengths = []
        orientations = []
        perimeters = []
        solidities = []

        s1_gen_props = sk.measure.regionprops(s1_glom_labels)
        for glom in range(s1_glom_count):

            s_label.append(1)
            glomID.append(glom+1)
            areas.append((s1_gen_props[glom].area)*df2*area_mpp2)
            bbs.append(s1_gen_props[glom].bbox)
            bb_areas.append((s1_gen_props[glom].bbox_area)*df2*area_mpp2)
            centroids.append(s1_gen_props[glom].centroid)
            convex_areas.append((s1_gen_props[glom].convex_area)*df2*area_mpp2)
            eccentricities.append(s1_gen_props[glom].eccentricity)
            equiv_diams.append((s1_gen_props[glom].equivalent_diameter)*df2*dist_mpp)
            extents.append(s1_gen_props[glom].extent)
            major_axis_lengths.append((s1_gen_props[glom].major_axis_length)*df2*dist_mpp)
            minor_axis_lengths.append((s1_gen_props[glom].minor_axis_length)*df2*dist_mpp)
            orientations.append(s1_gen_props[glom].orientation)
            perimeters.append((s1_gen_props[glom].perimeter)*df2*dist_mpp)
            solidities.append(s1_gen_props[glom].solidity)

        s2_gen_props = sk.measure.regionprops(s2_glom_labels)
        for glom in range(s2_glom_count):
            s_label.append(2)
            glomID.append(glom+1)
            areas.append((s2_gen_props[glom].area)*df2*area_mpp2)
            bbs.append(s2_gen_props[glom].bbox)
            bb_areas.append((s2_gen_props[glom].bbox_area)*df2*area_mpp2)
            centroids.append(s2_gen_props[glom].centroid)
            convex_areas.append((s2_gen_props[glom].convex_area)*df2*area_mpp2)
            eccentricities.append(s2_gen_props[glom].eccentricity)
            equiv_diams.append((s2_gen_props[glom].equivalent_diameter)*df2*dist_mpp)
            extents.append(s2_gen_props[glom].extent)
            major_axis_lengths.append((s2_gen_props[glom].major_axis_length)*df2*dist_mpp)
            minor_axis_lengths.append((s2_gen_props[glom].minor_axis_length)*df2*dist_mpp)
            orientations.append(s2_gen_props[glom].orientation)
            perimeters.append((s2_gen_props[glom].perimeter)*df2*dist_mpp)
            solidities.append(s2_gen_props[glom].solidity)

    elif num_sections == 1:

        se = sk.morphology.disk(3)
        pod_im = sk.morphology.binary_dilation(glom_image,selem=se,out=None)

        glom_label, glom_count = sp.ndimage.label(glom_image)
        total_gloms = glom_count

        glom_feat_array = []
        s_label = []
        glomID = []
        areas = []
        bbs = []
        bb_areas = []
        centroids = []
        convex_areas = []
        eccentricities = []
        equiv_diams = []
        extents = []
        major_axis_lengths = []
        minor_axis_lengths = []
        orientations = []
        perimeters = []
        solidities = []

        gen_props = sk.measure.regionprops(glom_label)
        for glom in range(glom_count):
            s_label.append(1)
            glomID.append(glom+1)
            areas.append((gen_props[glom].area)*df2*area_mpp2)
            bbs.append(gen_props[glom].bbox)
            bb_areas.append((gen_props[glom].bbox_area)*df2*area_mpp2)
            centroids.append(gen_props[glom].centroid)
            convex_areas.append((gen_props[glom].convex_area)*df2*area_mpp2)
            eccentricities.append(gen_props[glom].eccentricity)
            equiv_diams.append((gen_props[glom].equivalent_diameter)*df2*dist_mpp)
            extents.append(gen_props[glom].extent)
            major_axis_lengths.append((gen_props[glom].major_axis_length)*df2*dist_mpp)
            minor_axis_lengths.append((gen_props[glom].minor_axis_length)*df2*dist_mpp)
            orientations.append(gen_props[glom].orientation)
            perimeters.append((gen_props[glom].perimeter)*df2*dist_mpp)
            solidities.append(gen_props[glom].solidity)

    glom_feat_labels = ['section_label','glomID','glom_areas','glom_bb_areas','glom_convex_areas','glom_eccentricities','glom_equiv_diams','glom_extents','glom_major_axis_lengths','glom_minor_axis_lengths','glom_orientations','glom_perimeters','glom_solidities']
    glom_feat_qty = len(glom_feat_labels)

    glom_feat_array = np.hstack([np.array(s_label).reshape([total_gloms,1]),np.array(glomID).reshape([total_gloms,1]),
    np.array(areas).reshape([total_gloms,1]),np.array(bb_areas).reshape([total_gloms,1]),np.array(convex_areas).reshape([total_gloms,1]),
    np.array(eccentricities).reshape([total_gloms,1]),np.array(equiv_diams).reshape([total_gloms,1]),np.array(extents).reshape([total_gloms,1]),
    np.array(major_axis_lengths).reshape([total_gloms,1]),np.array(minor_axis_lengths).reshape([total_gloms,1]),np.array(orientations).reshape([total_gloms,1]),
    np.array(perimeters).reshape([total_gloms,1]),np.array(solidities).reshape([total_gloms,1])])

    return np.array(bbs), total_gloms, glom_feat_labels, glom_feat_qty, glom_feat_array

    print('features done')
