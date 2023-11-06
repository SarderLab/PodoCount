# PodoCount: A tool for whole-slide podocyte quantification (v1)

**Version 1.0.0**

This repository contains the source codes for the publication, "PodoCount: A robust, fully atuomated whole-slide podocyte quantification tool." All algorithms were developed and written by Briana Santo. The function xmltomask has been adapted from *Lutnick* et al.'s work "An integrated iterative annotation technique for easing neural network training in medical image analysis," *Nature Machine Intelligence*, 2019.

---
## A pre-print version of this work is available at: X

*Prepared by Briana Santo at SUNY Buffalo on 27July2021*

## Image Data

Whole slide images (WSIs) of murine kidney data are available at: http://bit.ly/3rdGPEd. Murine data includes whole kidney sections derived from both wild type and diseased mice across six mouse models [T2DM A, T2DM B, Aging, FSGS (SAND), HIVAN, Progeroid]. All kidney specimens were stained with p57kip2 immunohistochemistry and Periodic Acid-Schiff (without Hematoxylin counter stain) prior to digitization. 

## Requirements

This code runs using python3.

### Dependencies

- argparse [1.1]
- cv2 [4.1.2]
- lxml.etree [4.5.0]
- matplotlib [3.3.4]
- numpy [1.18.1]
- openslide-python [1.1.1]
- pandas [0.25.3]
- scikit-image [0.17.2]
- scipy [1.5.4]

### Modules from the Python Standard Library

- glob 
- os 
- sys
- time
- warnings

## Usage: 
### Running PodoCount from your own computer

The pipeline is run using: podocount_main_serv.py

Download the codes folder from GitHub titled either "PodoCount_Mouse_Analysis" or "PodoCount_Human_Analysis." Within the codes folder are two distinct subfolders entitled "WSIs" and "glom_xmls". Place WSIs for pipeline analysis in the "WSIs" folder; acceptable WSI formats include .svs and .ndpi. Place glomerulus annotation files in the "glom_xmls" folder. Glomerulus annotations (.xml files) may be generated through manual annotation or via our lab's H-AI-L tool; a convolutional neural network for glomerulus boundary detection developed by Lutnick et al. 

The code is run by using: podocount_main_serv.py

to run this code you must be in the "PodoCount_Mouse_Analysis" or "PodoCount_Human_Analysis" directory where it is contained, with WSIs and XMLs provided in the corresponding subfolders. 

Run the main script "podocount_main_serv.py", providing the necessary flags below:
- [--ftype] flag set to the WSI file extension
- [--slider] flag set to a value [0,3]
- [--cohort] set to the dataset or experiment name
- [--section_thickness] set to the tissue section thickness (an integer value within the range [1,15])
- [--num_sections] flag set to the number of tissue sections per slide (for WSIs of murine whole kidney sections options are 1 or 2; for human biopsy data, set to 1).

### For questions or feedback, please contact:
- Briana Santo <basanto@buffalo.edu>
- Pinaki Sarder <pinakisa@buffalo.edu>
