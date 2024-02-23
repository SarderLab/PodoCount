import os
import sys
from ctk_cli import CLIArgumentParser

def main(args):  
    
    cmd = "python3 ../PodoCount_code/podocount_main_serv.py  --type '{}' --input_image '{}' --basedir '{}' --slider {} --section_thickness {} --num_sections {} --ihc_gauss_sd {} --dt_gauss_sd {} --girderApiUrl {} --girderToken {} \
            ".format(args.type, args.input_image, args.basedir, args.slider, args.section_thickness, args.num_sections, args.ihc_gauss_sd, args.dt_gauss_sd, args.girderApiUrl, args.girderToken)
    
    print(cmd)
    sys.stdout.flush()
    os.system(cmd)  


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())