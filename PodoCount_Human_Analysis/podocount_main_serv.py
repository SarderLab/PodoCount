"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import openslide
import glob
import time
import warnings
import argparse
from xml_to_mask import xml_to_mask
from get_wsi_mask import get_wsi_mask
from get_boundary import get_boundary
from get_glom_props import get_glom_props
from get_pod_feat_spaces import get_pod_feat_spaces
from stain_decon import stain_decon
from get_pod_props import get_pod_props
from skimage import color, morphology
from skimage.transform import resize

#filter warnings
warnings.filterwarnings("ignore")

#parsing input arguments
parser = argparse.ArgumentParser(description='Parameters for operating PodoCount.')
parser.add_argument('-A','--ftype', metavar='', required = True, type=str, nargs='+',
                    help='the WSI file format; options include .svs and .ndpi')
parser.add_argument('-B','--num_sections', metavar='', required = True, type=int, nargs='+',
                    help='an integer indicating the number of tissue sections per WSI; options include {1,2}')
parser.add_argument('-C','--slider', metavar='', required = True, type=float, nargs='+',
                    help='a number (float) establishing the threshold on the dab stain; options include any number [0,3]')
parser.add_argument('-D','--section_thickness', metavar='', required = True, type=int, nargs='+',
                    help='a number (integer) indicating the tissue section thickness [0,15]')
parser.add_argument('-E','--cohort', metavar='', required = True, type=str, nargs='+',
                                        help='please indicate the name of the dataset')
args = parser.parse_args()

#get current working directory
cwd = os.getcwd()

#WSI general info
ftype = args.ftype[0]
section_thickness = args.section_thickness[0]
cohort_id = args.cohort[0]
WSIs = str(cwd + '/WSIs/*' + ftype)
glom_xmls = str(cwd + '/glom_xmls/*.xml')

WSI_dir = glob.glob(WSIs)
glom_xmls_dir = glob.glob(glom_xmls)

#Creating output directories
output_dir = str(cwd + '/output/' + cohort_id)
counters = str(output_dir + '/counters/')
contours = str(output_dir + '/contours/')
glom_feat_files = str(output_dir + '/glom_feat_files/')
pod_feat_files = str(output_dir + '/pod_feat_files/')
roi_dir = str(output_dir + '/roi_dir/')
pod_seg_dir = str(output_dir + '/pod_seg_dir/')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
    os.makedirs(counters)
    os.makedirs(contours)
    os.makedirs(glom_feat_files)
    os.makedirs(pod_feat_files)
    os.makedirs(roi_dir)
    os.makedirs(pod_seg_dir)

#Parameters
slider = args.slider[0]
num_sections = args.num_sections[0]

#Main Script
for WSI in range(len(WSI_dir)):
    start_time = time.time()

    WSI_file = WSI_dir[WSI]
    WSI_name = WSI_file.split('/')
    WSI_name = WSI_name[-1]
    WSI_name = WSI_name.split('.')
    WSI_name = WSI_name[0]
    print('\n')
    print('--- Working on: '+str(WSI_name)+' ---\n')
    WSI_file = openslide.open_slide(WSI_file)
    WSI_meta = (float(WSI_file.properties[openslide.PROPERTY_NAME_MPP_X])+float(WSI_file.properties[openslide.PROPERTY_NAME_MPP_Y]))/2
    dist_mpp, area_mpp2 = WSI_meta, WSI_meta**2
    WSI_cols,WSI_rows = WSI_file.dimensions[0],WSI_file.dimensions[1]
    if ftype == '.ndpi':
        level = 2
    else:
        level = 1
    WSI_levels = WSI_file.level_dimensions[level]
    downsample_factor=16
    df2 = np.sqrt(downsample_factor)

    #create output directory
    wsi_roi_dir = roi_dir + WSI_name
    wsi_pod_seg_dir = pod_seg_dir + WSI_name

    if not os.path.isdir(wsi_roi_dir):
        os.mkdir(wsi_roi_dir)
        os.mkdir(wsi_pod_seg_dir)

    #get whole-slide glom mask
    WSI_glom_xml = cwd + '/glom_xmls/' + WSI_name + '.xml'
    WSI_glom_mask = xml_to_mask(WSI_glom_xml, (0,0), (WSI_cols,WSI_rows), downsample_factor=downsample_factor, verbose=0)
    WSI_glom_mask = np.array(WSI_glom_mask)

    #get whole-slide mask
    WSI_downsample = np.array(WSI_file.read_region((0,0),level,(WSI_levels[0],WSI_levels[1])),dtype = 'uint8')
    WSI_downsample = get_wsi_mask(WSI_downsample[:,:,0:3],WSI_glom_mask)

    #xml files - initiation
    xml_counter = counters + WSI_name +'_counter.xml'
    xml_contour = contours + WSI_name + '.xml'

    xml_counter = open(xml_counter,'w')
    xml_contour = open(xml_contour,'w')

    xml_counter.write('<Annotations>\n\t<Annotation Id="1">\n\t\t<Attributes>\n\t\t\t<Attribute Id="0" Name="Pod" Value="" />\n\t\t</Attributes>\n\t\t<Regions>')
    xml_contour.write('<Annotations>\n\t<Annotation Id="1">\n\t\t<Attributes>\n\t\t\t<Attribute Id="0" Name="Pod" Value="" />\n\t\t</Attributes>\n\t\t<Regions>')

    gcount = 0
    pcount = 0

    #get ROI coordinates
    print('-- Step 1: Glomerulus localization and quantification --\n')
    bbs, total_gloms, glom_feat_labels, glom_feat_qty, glom_feat_array = get_glom_props(WSI_glom_mask,WSI_downsample,num_sections,dist_mpp,area_mpp2,df2)

    #define pod feature spaces
    glom_pod_feat_labels, glom_pod_feat_qty, glom_pod_feat_array, indv_pod_feat_labels, indv_pod_feat_qty, indv_pod_feat_array = get_pod_feat_spaces(total_gloms)

    print('-- Step 2: Podocyte detection and quantification --\n')
    for bb in range(len(bbs)):
        bb_iter = bb
        bb = bbs[bb]
        x_start,y_start,x_stop,y_stop = int(df2*bb[0]),int(df2*bb[1]),int(df2*bb[2]),int(df2*bb[3])
        x_length = x_stop-x_start
        y_length = y_stop-y_start

        roi = np.array(WSI_file.read_region((y_start,x_start),0,(y_length,x_length)),dtype = 'uint8')
        roi = roi[:,:,0:3]
        rows,cols,dims = roi.shape
        roi_name = '/roi_' + str(bb_iter+1) + '.png'
        roi_path = wsi_roi_dir + roi_name
        plt.imsave(roi_path,roi)

        glom_mask = WSI_glom_mask[bb[0]:bb[2],bb[1]:bb[3]]
        glom_mask = (resize(glom_mask,[rows,cols],anti_aliasing=True)>0.01)*1

        xml_counter, xml_contour, gcount, pcount, glom_pod_feat_vector, indv_pod_feats = get_pod_props(roi,glom_mask,slider,x_start,y_start,xml_counter,xml_contour, gcount, pcount,dist_mpp,area_mpp2, section_thickness, wsi_pod_seg_dir)
        glom_pod_feat_array[:,bb_iter] = glom_pod_feat_vector
        indv_pod_feat_array = np.vstack([indv_pod_feat_array,indv_pod_feats])

    #xml files - finalization
    xml_counter.write('\n\t\t</Regions>\n\t</Annotation>\n</Annotations>')
    xml_contour.write('\n\t\t</Regions>\n\t</Annotation>\n</Annotations>')

    xml_counter.close()
    xml_contour.close()

    print('-- Step 3: Feature file creation --\n')
    final_feat_array = np.vstack([glom_feat_array.T,glom_pod_feat_array])
    final_feat_labels = glom_feat_labels + glom_pod_feat_labels
    final_feat_DF = pd.DataFrame(final_feat_array,index=final_feat_labels)
    csv_path = glom_feat_files + WSI_name + '_Features.csv'
    final_feat_DF.to_csv(csv_path,index=True,columns=None)

    indv_pod_feat_array = np.delete(indv_pod_feat_array,(0),axis=0)
    final_feat_DF = pd.DataFrame(indv_pod_feat_array.T,index=indv_pod_feat_labels)
    csv_path = pod_feat_files + WSI_name + '_Pod_Features.csv'
    final_feat_DF.to_csv(csv_path,index=True,columns=None)

    print('--- Completed: '+str(WSI_name)+' ---\n')
    end_time = time.time() - start_time
    print("--- %s seconds for whole-slide analysis ---" % (end_time))

print('\n')
print('--- Completed full cohort ---')
print('\n')
