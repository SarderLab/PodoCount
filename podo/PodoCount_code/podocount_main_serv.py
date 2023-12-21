"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import pandas as pd
import os
import sys 
sys.path.append('..')
import time
import warnings
import argparse
from PIL import Image
from PodoCount_code.xml_to_mask import xml_to_mask
from PodoCount_code.get_wsi_mask import get_wsi_mask
from PodoCount_code.get_boundary import get_boundary
from PodoCount_code.get_glom_props import get_glom_props
from PodoCount_code.get_pod_feat_spaces import get_pod_feat_spaces
from PodoCount_code.stain_decon import stain_decon
from get_pod_props import get_pod_props
from get_pod_props_mouse import get_pod_props_mouse
from tiffslide import TiffSlide
import girder_client
from enum import Enum
import shutil
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

class InputType(Enum):
    Mouse_Analysis = 'Mouse'
    Human_Analysis = 'Human'

#filter warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--type')
parser.add_argument('--input_image')
parser.add_argument('--glom_xml')
parser.add_argument('--basedir')
parser.add_argument('--slider')
parser.add_argument('--section_thickness')
parser.add_argument('--num_sections')
parser.add_argument('--girderApiUrl')
parser.add_argument('--girderToken')

args = parser.parse_args()
gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
gc.setToken(args.girderToken)

folder = args.basedir
girder_folder_id = folder.split('/')[-2]
file_name = args.input_image.split('/')[-1]
files = list(gc.listItem(girder_folder_id))
item_dict = dict()

for file in files:
    d = {file['name']: file['_id']}
    item_dict.update(d)

slide_item_id = item_dict[file_name]
#get current working directory
cwd = os.getcwd()
WSIs = [args.input_image]

#WSI general info
section_thickness = int(args.section_thickness)
glom_xmls = [args.glom_xml]

#Creating output directories
output_dir = args.basedir + '/tmp'

# Check if the directory exists
if os.path.isdir(output_dir):
    # Remove the directory and its contents
    shutil.rmtree(output_dir)
# Create the directory
os.makedirs(output_dir)

#Parameters
slider = np.float64(args.slider)
num_sections = int(args.num_sections)

#Main Script
for WSI in WSIs:
    start_time = time.time()
    WSI_name = WSI.split('/')
    WSI_name = WSI[-1]
    WSI_name = WSI.split('.')
    WSI_name = WSI[0]
    print('\n')
    print('--- Working on: '+str(WSI_name)+' ---\n')

    WSI_file = TiffSlide(WSI)
    levels = WSI_file.level_dimensions

    mpp_x = 0.25
    mpp_y = 0.25
    WSI_meta = (mpp_x + mpp_y) / 2

    dist_mpp, area_mpp2 = WSI_meta, WSI_meta**2
    WSI_cols,WSI_rows = WSI_file.dimensions[0],WSI_file.dimensions[1]

    level = 0

    if (len(levels) > 1):
        level = 1
    else:
        level = 0

    WSI_levels = WSI_file.level_dimensions[level]

    downsample_factor=16
    df2 = np.sqrt(downsample_factor)

    #get whole-slide glom mask
    WSI_glom_xml = args.glom_xml
    WSI_glom_mask = xml_to_mask(WSI_glom_xml, (0,0), (WSI_cols,WSI_rows), downsample_factor=downsample_factor, verbose=0)
    WSI_glom_mask = np.array(WSI_glom_mask)

    #get whole-slide mask
    WSI_downsample = np.asarray(WSI_file.read_region((0,0),level, levels[level]),dtype = 'uint8')

    WSI_downsample = get_wsi_mask(WSI_downsample[:,:,0:3],WSI_glom_mask)

    #xml files - initiation
    xml_counter = output_dir + WSI_name +'_counter.xml'
    xml_contour = output_dir + WSI_name + '_contour.xml'
    xml_counter_path = output_dir + WSI_name +'_counter.xml'
    xml_contour_path = output_dir + WSI_name + '_contour.xml'

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
        glom_mask = WSI_glom_mask[bb[0]:bb[2],bb[1]:bb[3]]
        glom_mask_uint8 = (glom_mask * 255).astype(np.uint8)
        pil_image_glom = Image.fromarray(glom_mask_uint8)
        width, height = glom_mask.shape
        glom_mask = pil_image_glom.resize((height, width), Image.Resampling.LANCZOS)
        glom_mask = np.array(glom_mask)
        glom_mask = glom_mask // 255

        if args.type == InputType.Human_Analysis.value:
            xml_counter, xml_contour, gcount, pcount, glom_pod_feat_vector, indv_pod_feats = get_pod_props(roi,glom_mask,slider,x_start,y_start,xml_counter,xml_contour, gcount, pcount,dist_mpp,area_mpp2, section_thickness)
        elif args.type == InputType.Mouse_Analysis.value:
            xml_counter, xml_contour, gcount, pcount, glom_pod_feat_vector, indv_pod_feats = get_pod_props_mouse(roi,glom_mask,slider,x_start,y_start,xml_counter,xml_contour, gcount, pcount,dist_mpp,area_mpp2, section_thickness)
            
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
    csv_path_glom = output_dir + WSI_name + '_Glom_Features.csv'
    final_feat_DF.to_csv(csv_path_glom,index=True,columns=None)

    indv_pod_feat_array = np.delete(indv_pod_feat_array,(0),axis=0)
    final_feat_DF = pd.DataFrame(indv_pod_feat_array.T,index=indv_pod_feat_labels)
    csv_path_pod = output_dir + WSI_name + '_Pod_Features.csv'
    final_feat_DF.to_csv(csv_path_pod,index=True,columns=None)

    print('--- Completed: '+str(WSI_name)+' ---\n')
    end_time = time.time() - start_time
    print("--- %s seconds for whole-slide analysis ---" % (end_time))

    gc.uploadFileToItem(slide_item_id, csv_path_glom, reference=None, mimeType=None, filename=None, progressCallback=None)
    gc.uploadFileToItem(slide_item_id, csv_path_pod, reference=None, mimeType=None, filename=None, progressCallback=None)
    #gc.uploadFileToItem(slide_item_id, xml_counter_path, reference=None, mimeType=None, filename=None, progressCallback=None)
    #gc.uploadFileToItem(slide_item_id, xml_contour_path, reference=None, mimeType=None, filename=None, progressCallback=None)

shutil.rmtree(output_dir)

print('\n')
print('--- Completed full cohort ---')
print('\n')
