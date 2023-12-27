import torch, numpy as np
import torch.nn as nn
import imageio
import sys, getopt
import os
import glob
from topohpm.scripts.utils import zoom, load_itk, save_itk
from skimage.morphology import remove_small_holes, remove_small_objects

def f(input_path, output_path, prob_threshold = 0.5, area_threshold = 400, zoom_size = None, dim = 2):
    if dim == 2:
        input = np.asarray(imageio.imread(input_path)).astype(float)
        input /= 255.0
    else:
        input, _, _ = load_itk(input_path)
    if zoom_size is not None:
        input = zoom(input, target_shape = zoom_size)
        
    input = remove_small_holes(input > prob_threshold, area_threshold = area_threshold / 2, connectivity = 2).astype(float)
    # remove noisy and disconnected parts
    input = remove_small_objects(input > prob_threshold, min_size = area_threshold, connectivity = 2).astype(float)
    print('write to '+output_path)
    if dim == 2:
        imageio.imwrite(output_path, (input * 255).astype('uint8'))
    else:
        save_itk(output_path, input.astype(float))

def main(argv):
    input_path = ''
    output_path = ''
    prob_threshold = 0.5
    area_threshold = 400
    dim = 2
    opts, args = getopt.getopt(argv, "hi:o:p:a:d:z:", ['help', 'input=', 'output=', 'pthreshold=' ,'athreshold=', 'zoom=', 'dim='])
    zoom_size = None
    if len(opts) == 0:
        print('unknow options, usage: evaluate.py -i <input_file> -o <output_file> -p <prob_threshold = 0.5> -a <area_threshold = 400> -z <zoom = None> -d <dim = 2>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: evaluate.py -i <input_file> -o <output_file> -p <prob_threshold = 0.5> -a <area_threshold = 400> -z <zoom = None> -d <dim = 2>')
            sys.exit()
        elif opt in ("-i", '--input'):
            input_path = arg
        elif opt in ("-o", '--output'):
            output_path = arg
        elif opt in ("-p", '--pthreshold'):
            prob_threshold = float(arg)
        elif opt in ("-a", '--athreshold'):
            area_threshold = int(arg)
        elif opt in ('-z', '--zoom'):
            zoom_size = tuple([int(z) for z in arg.split(',')])
        elif opt in ('-d', '--dim'):
            dim = int(arg)
        else:
            print('unknow option, usage: evaluate.py -i <input_file> -o <output_file> -p <prob_threshold = 0.5> -a <area_threshold = 400> -z <zoom = None> -d <dim = 2>')
            sys.exit()

    if os.path.isdir(input_path):
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        input_paths = sorted(glob.glob(os.path.join(input_path, '*')))
        for ifile in input_paths:
            _, filename = os.path.split(ifile)
            filename, _ = os.path.splitext(filename)
            filename, _ = os.path.splitext(filename)
            ofile = output_path+'/'+ filename + '.png'
            if dim == 3:
                ofile = output_path+'/'+ filename + '.nii.gz'
            f(ifile, ofile, prob_threshold, area_threshold, zoom_size, dim)
        
    else:
        res = f(input_path, output_path, prob_threshold, area_threshold, zoom_size, dim)
if __name__ == "__main__":
    main(sys.argv[1:])