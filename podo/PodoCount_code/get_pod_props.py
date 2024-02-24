"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
import scipy as sp
from stain_decon import stain_decon
from get_boundary import get_boundary

#Functions
def thin_section_method(podocyte_count,pod_mask,areas,bbs,glom_area,T,area_mpp2,dist_mpp):

    bbs = np.array(bbs).reshape([podocyte_count,4])
    x_start,y_start,x_stop,y_stop = bbs[:,0],bbs[:,1],bbs[:,2],bbs[:,3]
    x_lengths = dist_mpp*(x_stop-x_start)
    y_lengths = dist_mpp*(y_stop-y_start)
    d = np.mean(np.mean(np.hstack([x_lengths,y_lengths]),axis=0))
    k = 0.72
    D = (d-T+np.sqrt((d-T)**2+(4*k*d*T)))/(2*k)
    CF = 1/(D/T+1)
    thin_pod_count = podocyte_count*CF
    glom_vol = glom_area*T
    thin_pod_density = thin_pod_count/glom_vol
    #swapped in zeros
    thin_pod_tpa = 0
    thin_pod_gpc = 0
    thin_pod_mask = np.zeros(pod_mask.shape)
    return thin_pod_count, thin_pod_density, thin_pod_tpa, thin_pod_gpc, thin_pod_mask

def get_stats(array):
    array = np.array(array)
    mean = np.mean(array)
    std = np.std(array)
    median = np.quantile(array,0.5)
    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)
    min = np.min(array)
    max = np.max(array)
    stats = np.hstack([mean, std, median, q1, q3, min, max])
    return stats


def get_pod_props(roi,glom_mask,slider,x_start,y_start,xml_counter,xml_contour, gcount, pcount,dist_mpp,area_mpp2, section_thickness, ihc_gauss_sd, dt_gauss_sd):
    #Parameters
    #ihc_gauss_sd = 2
    #dt_gauss_sd = 1
    min_area = 200
    max_major = 60
    min_minor = 10
    max_ecc = 0.7
    emt_thresh = 0.01

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

    nuclei_dt = sp.ndimage.morphology.distance_transform_edt(np.invert(nuc_temp.astype(np.uint8)))
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
    glom_mask = sk.morphology.binary_erosion(glom_mask,footprint=se,out=None)
    

    separated_podocytes = np.logical_and(glom_mask,separated_podocytes)

    
    ###human add end
    gcount = gcount+1

    ###PODOCYTE FEATURE ENGINEERING AND EXTRACTION###
    podocyte_label, podocyte_count = sp.ndimage.label(separated_podocytes)

    if podocyte_count>0:

        #revise pod mask to thin section method count
        gen_props = sk.measure.regionprops(podocyte_label,ihc)
        areas = []
        bbs = []
        for pod in range(podocyte_count):
            areas.append(gen_props[pod].area)
            bbs.append(gen_props[pod].bbox)
        thin_pod_count, thin_pod_density, thin_pod_tpa, thin_pod_gpc, thin_pod_mask = thin_section_method(podocyte_count,separated_podocytes,areas,bbs,glom_area,section_thickness,area_mpp2,dist_mpp)

    if podocyte_count>0:

        glom_pod_feat_vector = []
        indv_pod_feats = []
        gcounts = gcount*np.ones([1,podocyte_count])
        pcounts = []
        local_pcounts = []
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
        bowmans_dists = []
        glom_ctr_dists = []

        total_pod_area = np.sum(separated_podocytes)*area_mpp2
        pod_glom_area = total_pod_area/glom_area

        gen_props = sk.measure.regionprops(podocyte_label,ihc)
        glom_boundary = get_boundary(glom_mask)

        for pod in range(podocyte_count):

            pcount = pcount+1
            pcounts.append(pcount)
            local_pcounts.append(pod+1)
            areas.append(gen_props[pod].area*area_mpp2)
            centroids.append(gen_props[pod].centroid)
            major_axis_lengths.append(gen_props[pod].major_axis_length*dist_mpp)
            bb_areas.append(gen_props[pod].bbox_area*area_mpp2)
            convex_areas.append(gen_props[pod].convex_area*area_mpp2)
            eccentricities.append(gen_props[pod].eccentricity)
            equiv_diams.append(gen_props[pod].equivalent_diameter)
            extents.append(gen_props[pod].extent)
            minor_axis_lengths.append(gen_props[pod].minor_axis_length*dist_mpp)
            max_intensities.append(gen_props[pod].max_intensity)
            mean_intensities.append(gen_props[pod].mean_intensity)
            min_intensities.append(gen_props[pod].min_intensity)
            orientations.append(gen_props[pod].orientation)
            perimeters.append(gen_props[pod].perimeter*dist_mpp)
            solidities.append(gen_props[pod].solidity)

            #find bowmans and glom center distances
            centroid = np.array(centroids[pod]).reshape(1,-1)
            distances = sp.spatial.distance.cdist(centroid, glom_boundary, 'euclidean')
            bowmans_dists.append(np.min(distances)*dist_mpp)

            glom_label, glom_count = sp.ndimage.label(glom_mask)
            glom_centroid = sk.measure.regionprops(glom_label)
            glom_centroid = np.array(glom_centroid[0].centroid).reshape(1,-1)
            distances = sp.spatial.distance.cdist(centroid,glom_centroid, 'euclidean')
            glom_ctr_dists.append(np.min(distances)*dist_mpp)

        #find podocyte spatial density
        pod_dists = sp.spatial.distance.pdist(centroids)
        inter_pod_dist = sum(pod_dists)*dist_mpp
        if podocyte_count>1:
            inter_pod_dist = 1/inter_pod_dist
        else:
            inter_pod_dist = 0

        indv_pod_feats = np.vstack([gcounts,pcounts,local_pcounts,areas,bb_areas,convex_areas,eccentricities,equiv_diams,extents,major_axis_lengths,minor_axis_lengths,max_intensities,mean_intensities,min_intensities,orientations,perimeters,solidities,bowmans_dists,glom_ctr_dists])
        indv_pod_feats = np.vstack(indv_pod_feats).T

        glom_pod_feat_vector.append(podocyte_count)
        glom_pod_feat_vector.append(thin_pod_count)
        glom_pod_feat_vector.append(thin_pod_density)
        glom_pod_feat_vector.append(thin_pod_tpa)
        glom_pod_feat_vector.append(thin_pod_gpc)
        glom_pod_feat_vector.append(total_pod_area)
        glom_pod_feat_vector.append(pod_glom_area)
        glom_pod_feat_vector.append(inter_pod_dist)
        glom_pod_feat_vector.append(get_stats(areas))
        glom_pod_feat_vector.append(get_stats(bb_areas))
        glom_pod_feat_vector.append(get_stats(convex_areas))
        glom_pod_feat_vector.append(get_stats(eccentricities))
        glom_pod_feat_vector.append(get_stats(equiv_diams))
        glom_pod_feat_vector.append(get_stats(extents))
        glom_pod_feat_vector.append(get_stats(major_axis_lengths))
        glom_pod_feat_vector.append(get_stats(minor_axis_lengths))
        glom_pod_feat_vector.append(get_stats(max_intensities))
        glom_pod_feat_vector.append(get_stats(mean_intensities))
        glom_pod_feat_vector.append(get_stats(min_intensities))
        glom_pod_feat_vector.append(get_stats(orientations))
        glom_pod_feat_vector.append(get_stats(perimeters))
        glom_pod_feat_vector.append(get_stats(solidities))
        glom_pod_feat_vector.append(get_stats(bowmans_dists))
        glom_pod_feat_vector.append(get_stats(glom_ctr_dists))

        glom_pod_feat_vector = np.hstack(glom_pod_feat_vector)

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
                pod_im = sk.morphology.binary_dilation(pod_im,footprint=se,out=None)
                pod_boundary = get_boundary(pod_im)

                L = []
                xml_regionID = str(pod)
                index = pod-1
                length = major_axis_lengths[index]
                area = areas[index]
                length_um = length*dist_mpp
                area_um = area*area_mpp2
                xml_contour.write('\n\t\t\t<Region Id="' + str(pcount) + '" Type="0" Zoom="0" Selected="0" ImageLocation="" ImageFocus="-1" Length="' + str(length) + '" Area="'+ str(area) +'" LengthMicrons="'+ str(length_um) +'" AreaMicrons="'+ str(area_um) +'" Text="" NegativeROA="0" InputRegionId="0" Analyze="0" DisplayId="1">\n\t\t\t\t<Attributes/>\n\t\t\t\t<Vertices>\n')
                for point in pod_boundary:
                    xml_Y = str((point[1]+x_start))
                    xml_X = str((point[0]+y_start))
                    L.append(str('\t\t\t\t\t<Vertex X="' + xml_X + '" Y="' + xml_Y + '" Z="0"/>\n'))
                xml_contour.writelines(L)
                xml_contour.write('\t\t\t\t</Vertices>\n\t\t\t</Region>')

        glom_pod_feat_vector = np.array(glom_pod_feat_vector)

    elif podocyte_count==0:
        glom_pod_feat_vector = np.zeros([120,])
        indv_pod_feats = np.zeros([1,19])
        indv_pod_feats[0,0] = gcount

    return xml_counter, xml_contour, gcount, pcount, glom_pod_feat_vector, indv_pod_feats
