from utils import *
import sys, getopt
import numpy as np 
import os
import glob
def f(inputfile, outputfile, scaling, target_shape):
    img_array, origin, spacing = load_itk(inputfile)
    result_array, start_idx = crop_middle(img_array, scaling = scaling, target_shape = target_shape)
    origin = origin + start_idx.astype(float) * spacing
    print('from', str(img_array.shape), 'to', str(result_array.shape))
    print('write to', outputfile)
    save_itk(outputfile, result_array, origin = origin, spacing = spacing)
def main(argv):
    inputfile = ''
    outputfile = ''
    scaling = (0.5, 0.5, 0.5)
    target_shape = None
    opts, args = getopt.getopt(argv, "hi:o:s:t:", ['help', 'input=', 'output=', 'scaling=', 'target_shape='])
    if len(opts) == 0:
        print('unknow options, usage: crop_middle.py -i <input_prob_volume> -o <output_prob_volume> -s <scaling = 0.5,0.5,0.5>  -t <target_shape>. \n Note that you only need to specify scaling factor or target shape.')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: crop_middle.py -i <input_prob_volume> -o <output_prob_volume> -s <scaling = 0.5,0.5,0.5>  -t <target_shape>. \n Note that you only need to specify scaling factor or target shape.')
            sys.exit()
        elif opt in ("-i", '--input'):
            inputfile = arg
        elif opt in ("-o", '--output'):
            outputfile = arg
        elif opt in ("-s", '--scaling'):
            scaling = tuple(float(s) for s in arg.split(','))
            if len(scaling) != 3:
                print('Error: The length of scaling factor need to be 3. Example: -s 0.5,0.5,0.5')
                sys.exit() 
        elif opt in ('-t', '--target_shape'):
            target_shape = tuple(int(ts) for ts in arg.split(','))
            if len(target_shape) != 3:
                print('Error: The length of target_shape need to be 3. Example: -t 100,200,200')
                sys.exit()             
        else:
            print('usage: crop_middle.py -i <input_prob_volume> -o <output_prob_volume> -s <scaling = 0.5,0.5,0.5>  -t <target_shape>. \n Note that you only need to specify scaling factor or target shape.')
            sys.exit()
    if os.path.isdir(inputfile):
        input_files = sorted(glob.glob(os.path.join(inputfile, '*.nii.gz')))
        for ifile in input_files:
            _, filename = os.path.split(ifile)
            filename, ext = os.path.splitext(filename)
            filename, ext0 = os.path.splitext(filename)
            if ext0 != '':
                ext = ext0 + ext

            ofile = outputfile+'/'+ filename + '_cropped' + ext 
            f(ifile, ofile, scaling, target_shape)

    else:
        f(inputfile, outputfile, scaling, target_shape)
if __name__ == "__main__":
    main(sys.argv[1:])