"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np

def get_pod_feat_spaces(total_gloms):

    glom_pod_feat_labels = ['raw_pod_cts','thin_pod_cts','thin_pod_density','thin_pod_tpa','thin_pod_gpc','total_pod_area','pod_glom_area_ratio','inter_pod_dist','mean_pod_areas','std_pod_areas','med_pods_areas','q1_pod_areas','q3_pod_areas','min_pod_areas','max_pod_areas','mean_pod_bb_areas','std_pod_bb_areas','med_pod_bb_areas','q1_pod_bb_areas','q3_pod_bb_areas','min_pod_bb_areas','max_pod_bb_areas','mean_pod_convex_areas','std_pod_convex_areas','med_pod_convex_areas','q1_pod_convex_areas','q3_pod_convex_areas','min_pod_convex_areas','max_pod_convex_areas','mean_pod_eccentricities','std_pod_eccentricities','med_pod_eccentricities','q1_pod_eccentricities','q3_pod_eccentricities','min_pod_eccentricities','max_pod_eccentricities','mean_pod_equiv_diams','std_pod_equiv_diams','med_pod_equiv_diams','q1_pod_equiv_diams','q3_pod_equiv_diams','min_pod_equiv_diams','max_pod_equiv_diams','mean_pod_extents','std_pod_extents','med_pod_extents','q1_pod_extents','q3_pod_extents','min_pod_extents','max_pod_extents','mean_pod_major_axis_lengths','std_pod_major_axis_lengths','med_pod_major_axis_lengths','q1_pod_major_axis_lengths','q3_pod_major_axis_lengths','min_pod_major_axis_lengths','max_pod_major_axis_lengths','mean_pod_minor_axis_lengths','std_pod_minor_axis_lengths','med_pod_minor_axis_lengths','q1_pod_minor_axis_lengths','q3_pod_minor_axis_lengths','min_pod_minor_axis_lengths','max_pod_minor_axis_lengths','mean_pod_max_intensities','std_pod_max_intensities','med_pod_max_intensities','q1_pod_max_intensities','q3_pod_max_intensities','min_pod_max_intensities','max_pod_max_intensities','mean_pod_mean_intensities','std_pod_mean_intensities','med_pod_mean_intensities','q1_pod_mean_intensities','q3_pod_mean_intensities','min_pod_mean_intensities','max_pod_mean_intensities','mean_pod_min_intensities','std_pod_min_intensities','med_pod_min_intensities','q1_pod_min_intensities','q3_pod_min_intensities','min_pod_min_intensities','max_pod_min_intensities','mean_pod_orientations','std_pod_orientations','med_pod_orientations','q1_pod_orientations','q3_pod_orientations','min_pod_orientations','max_pod_orientations','mean_pod_perimeters','std_pod_perimeters','med_pod_perimeters','q1_pod_perimeters','q3_pod_perimeters','min_pod_perimeters','max_pod_perimeters','mean_pod_solidities','std_pod_solidities','med_pod_solidities','q1_pod_solidities','q3_pod_solidities','min_pod_solidities','max_pod_solidities','mean_pod_bowmans_dists','std_pod_bowmans_dists','med_pod_bowmans_dists','q1_pod_bowmans_dists','q3_pod_bowmans_dists','min_pod_bowmans_dists','max_pod_bowmans_dists','mean_pod_glom_ctr_dists','std_pod_glom_ctr_dists','med_pod_glom_ctr_dists','q1_pod_glom_ctr_dists','q3_pod_glom_ctr_dists','min_pod_glom_ctr_dists','max_pod_glom_ctr_dists']

    glom_pod_feat_qty = len(glom_pod_feat_labels)
    glom_pod_feat_array = np.empty([glom_pod_feat_qty,total_gloms])

    indv_pod_feat_labels = ['glom_id','global_pod_id','local_pod_id','pod_areas','pod_bb_areas','pod_convex_areas','pod_eccentricities','pod_equiv_diams','pod_extents','pod_major_axis_lengths','pod_minor_axis_lengths','pod_max_intensities','pod_mean_intensities','pod_min_intensities','pod_orientations','pod_perimeters','pod_solidities','pod_bowmans_dist','pod_glom_ctr_dist']

    indv_pod_feat_qty = len(indv_pod_feat_labels)
    indv_pod_feat_array = np.zeros([1,indv_pod_feat_qty])

    return glom_pod_feat_labels, glom_pod_feat_qty, glom_pod_feat_array, indv_pod_feat_labels, indv_pod_feat_qty, indv_pod_feat_array
