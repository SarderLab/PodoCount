"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
import scipy as sp
import cv2
from skimage import segmentation
from stain_decon import stain_decon
from get_boundary import get_boundary

#Functions
def get_pod_props(roi,glom_mask,slider,x_start,y_start,xml_counter,xml_contour,count,dist_mpp,area_mpp2, ihc_gauss_sd, dt_gauss_sd, max_major, min_minor, max_ecc, emt_thresh):
    #Parameters
    #ihc_gauss_sd = 2
    #dt_gauss_sd = 1
    min_area = 200
    #max_major = 60
    #min_minor = 10
    #max_ecc = 0.7
    #emt_thresh = 0.01

    ###PODOCYTE SEGMENTATION###
    res,pas,ihc = stain_decon(roi)

    ihc = 1-pas
    ihc = np.uint8((sk.filters.gaussian(ihc,ihc_gauss_sd)*255))
    ihc_mean = ihc.mean()
    ihc_std = ihc.std()
    ihc_thresh = (ihc_mean + (slider*ihc_std))

    nuclei = ihc>ihc_thresh
    nuclei = sp.ndimage.morphology.binary_fill_holes(nuclei)
    nuclei = sk.morphology.remove_small_objects(nuclei, min_size=100, connectivity=8)

    nuclei_label, num_labels = sp.ndimage.label(nuclei)
    nuclei_props = sk.measure.regionprops(nuclei_label)
    nuc_temp = np.zeros(nuclei.shape)
    for nuc in range(num_labels):
        label = nuc + 1
        nuc_major = nuclei_props[nuc].major_axis_length
        nuc_minor = nuclei_props[nuc].minor_axis_length
        nuc_ecc = nuclei_props[nuc].eccentricity
        if nuc_major < max_major:
            if nuc_minor > min_minor:
                nuc_temp[nuclei_label==label] = 1
        elif nuc_ecc < max_ecc:
            if nuc_minor > min_minor:
                nuc_temp[nuclei_label==label] = 1
        elif nuc_ecc < max_ecc:
            nuc_temp[nuclei_label==label] = 1

    nuclei_dt = sp.ndimage.morphology.distance_transform_edt(np.invert(nuc_temp.astype(np.int)))
    nuclei_dt = sk.filters.gaussian(nuclei_dt,dt_gauss_sd)
    nuclei_dt_max = sk.morphology.h_minima(nuclei_dt,emt_thresh*nuclei_dt.max())

    nuclei_label, num_labels = sp.ndimage.label(nuc_temp - nuclei_dt_max)
    nuclei_props = sk.measure.regionprops(nuclei_label)
    nuc_euls = []

    for nuc in range(num_labels):
        label = nuclei_label[nuc] + 1
        nuc_eul = nuclei_props[nuc].euler_number
        if nuc_eul <0:
            nuc_euls.append(nuc)

    nuclei_multi = np.zeros(nuclei_label.shape)
    if np.sum(nuc_euls)>0:
        for nuc in range(len(nuc_euls)):
            label = nuc_euls[nuc] + 1
            nuclei_multi[nuclei_label==label] = 1

        nuclei_multi = sp.ndimage.morphology.binary_fill_holes(nuclei_multi)
        nuc_temp[nuc_temp==nuclei_multi] = 0
        nuclei_multi_dt = sp.ndimage.morphology.distance_transform_edt(nuclei_multi)
        markers = sp.ndimage.label(nuclei_multi_dt[:,:]>10)[0]
        separated_podocytes = sk.segmentation.watershed(-1*nuclei_multi_dt[:,:],markers,mask=nuclei_multi_dt[:,:]>0,watershed_line=True)
        separated_podocytes = separated_podocytes>0
        separated_podocytes = sk.morphology.remove_small_objects(separated_podocytes, min_size=100, connectivity=8)
        separated_podocytes[nuc_temp==1]=1

    else:
        separated_podocytes = nuc_temp
        markers = np.zeros(nuclei_label.shape)

    rows,cols = np.shape(separated_podocytes)
    glom_mask = np.resize(glom_mask,(rows,cols))
    glom_area = np.sum(glom_mask)*area_mpp2
    ###human add
    se = sk.morphology.disk(3)
    glom_mask = sk.morphology.binary_erosion(glom_mask,selem=se,out=None)
    separated_podocytes = np.logical_and(glom_mask,separated_podocytes)
    ###human add end

    ###PODOCYTE FEATURE ENGINEERING AND EXTRACTION###
    podocyte_label, podocyte_count = sp.ndimage.label(separated_podocytes)

    if podocyte_count>0:

        pod_feat_vector = []
        areas = []
        centroids = []
        major_axis_lengths = []

        bb_areas = []
        convex_areas = []
        eccentricities = []
        equiv_diams = []
        extents = []
        minor_axis_lengths = []
        max_intensities = []
        mean_intensities = []
        min_intensities = []
        orientations = []
        perimeters = []
        solidities = []

        total_pod_area = np.sum(separated_podocytes)*area_mpp2
        pod_per_glom = podocyte_count/glom_area
        pod_glom_area = total_pod_area/glom_area

        gen_props = sk.measure.regionprops(podocyte_label,ihc)
        for pod in range(podocyte_count):

            areas.append(gen_props[pod].area)
            centroids.append(gen_props[pod].centroid)
            major_axis_lengths.append(gen_props[pod].major_axis_length)

            bb_areas.append(gen_props[pod].bbox_area)
            convex_areas.append(gen_props[pod].convex_area)
            eccentricities.append(gen_props[pod].eccentricity)
            equiv_diams.append(gen_props[pod].equivalent_diameter)
            extents.append(gen_props[pod].extent)
            minor_axis_lengths.append(gen_props[pod].minor_axis_length)
            max_intensities.append(gen_props[pod].max_intensity)
            mean_intensities.append(gen_props[pod].mean_intensity)
            min_intensities.append(gen_props[pod].min_intensity)
            orientations.append(gen_props[pod].orientation)
            perimeters.append(gen_props[pod].perimeter)
            solidities.append(gen_props[pod].solidity)

        pod_feat_vector.append(podocyte_count)
        pod_feat_vector.append(pod_per_glom)
        pod_feat_vector.append(total_pod_area)
        pod_feat_vector.append(pod_glom_area)
        area = np.mean(np.array(areas))*area_mpp2
        pod_feat_vector.append(area)
        bb_area = np.mean(np.array(bb_areas))*area_mpp2
        pod_feat_vector.append(bb_area)
        convex_area = np.mean(np.array(convex_areas))*area_mpp2
        pod_feat_vector.append(convex_area)
        eccentricity = np.mean(np.array(eccentricities))
        pod_feat_vector.append(eccentricity)
        equiv_diam = np.mean(np.array(equiv_diams))*dist_mpp
        pod_feat_vector.append(equiv_diam)
        extent = np.mean(np.array(extents))
        pod_feat_vector.append(extent)
        major_axis_length = np.mean(np.array(major_axis_lengths))*dist_mpp
        pod_feat_vector.append(major_axis_length)
        minor_axis_length = np.mean(np.array(minor_axis_lengths))*dist_mpp
        pod_feat_vector.append(minor_axis_length)
        max_intensity = np.mean(np.array(max_intensities))
        pod_feat_vector.append(max_intensity)
        mean_intensity = np.mean(np.array(mean_intensities))
        pod_feat_vector.append(mean_intensity)
        min_intensity = np.mean(np.array(min_intensities))
        pod_feat_vector.append(min_intensity)
        orientation = np.mean(np.array(orientations))
        pod_feat_vector.append(orientation)
        perimeter = np.mean(np.array(perimeters))*dist_mpp
        pod_feat_vector.append(perimeter)
        solidity = np.mean(np.array(solidities))
        pod_feat_vector.append(solidity)

        #write pod xml files
        #counter tool
        for pod in range(podocyte_count):
            centroid = np.array(centroids[pod]).reshape(-1,1)
            xml_regionID = str(pod + 1)
            xml_Y = str(round(centroid[0][0]+x_start))
            xml_X = str(round(centroid[1][0]+y_start))

            xml_counter.write('\n\t\t\t<Region Id="' + xml_regionID + '" Type="5">\n\t\t\t\t<Vertices>\n\t\t\t\t\t<Vertex X="' + xml_X + '" Y="' + xml_Y + '" Z="0" />\n\t\t\t\t</Vertices>\n\t\t\t</Region>')

        #contours
        for pod in range(podocyte_count+1):
            if pod>0:
                pod_im = np.zeros(separated_podocytes.shape)
                pod_im[podocyte_label==(pod)] = 1
                se = sk.morphology.disk(2)
                pod_im = sk.morphology.binary_dilation(pod_im,selem=se,out=None)
                pod_boundary = get_boundary(pod_im)

                L = []
                count = count+1
                xml_regionID = str(pod)
                index = pod-1
                length = major_axis_lengths[index]
                area = areas[index]
                length_um = length*dist_mpp
                area_um = area*area_mpp2
                xml_contour.write('\n\t\t\t<Region Id="' + str(count) + '" Type="0" Zoom="0" Selected="0" ImageLocation="" ImageFocus="-1" Length="' + str(length) + '" Area="'+ str(area) +'" LengthMicrons="'+ str(length_um) +'" AreaMicrons="'+ str(area_um) +'" Text="" NegativeROA="0" InputRegionId="0" Analyze="0" DisplayId="1">\n\t\t\t\t<Attributes/>\n\t\t\t\t<Vertices>\n')
                for point in pod_boundary:
                    xml_Y = str((point[1]+x_start))
                    xml_X = str((point[0]+y_start))
                    L.append(str('\t\t\t\t\t<Vertex X="' + xml_X + '" Y="' + xml_Y + '" Z="0"/>\n'))
                xml_contour.writelines(L)
                xml_contour.write('\t\t\t\t</Vertices>\n\t\t\t</Region>')

        pod_feat_vector = np.array(pod_feat_vector)

    elif podocyte_count==0:
        pod_feat_vector = np.zeros([18,])

    return xml_counter, xml_contour, count, pod_feat_vector