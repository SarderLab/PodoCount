import os
import sys
from ctk_cli import CLIArgumentParser

def main(args):  
    
    cmd = "python3 ../PodoCount_code/podocount_main_serv.py  --input_image '{}' --glom_xml '{}' --basedir '{}' --slider {}  --section_thickness {} --girderApiUrl {} --girderToken {} \
            ".format(args.input_image, args.glom_xml, args.basedir, args.slider, args.section_thickness, args.girderApiUrl, args.girderToken)
    
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)  

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())