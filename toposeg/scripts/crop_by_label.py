from utils import *
import sys, getopt
import numpy as np 
import os
import glob
def f(inputdata, inputlabel, outputdata, outputlabel, margin, background):
    print('read {} and {}'.format(inputdata, inputlabel))
    data_array, origin, spacing = load_itk(inputdata)
    label_array, _, _ = load_itk(inputlabel)
    # print('???')
    res_label_array, res_data_array, start_idx = crop_by_boundingbox(label = label_array, data = data_array, margin = margin, background = background)
    origin = origin + start_idx.astype(float) * spacing
    print('from', str(data_array.shape), 'to', str(res_data_array.shape))
    print('write to', outputdata, outputlabel)
    save_itk(outputdata, res_data_array, origin = origin, spacing = spacing)
    save_itk(outputlabel, res_label_array, origin = origin, spacing = spacing)
def main(argv):
    inputdata = ''
    inputlabel = ''

    outputdata = ''
    outputlabel = ''
    margin = (20, 20, 20)
    background = 0.0

    opts, args = getopt.getopt(argv, "hm:b:", ['help', 'id=', 'il=', 'od=', 'ol=', 'margin=', 'background='])
    if len(opts) == 0:
        print('unknow options, usage: crop_by_label.py --id <input_data_file> --il <input_label_file> --od <output_data_file> --ol <output_data_file> -m <margin = 20,20,20>  -b <background = 0.0>.')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: crop_by_label.py --id <input_data_file> --il <input_label_file> --od <output_data_file> --ol <output_data_file> -m <margin = 20,20,20>  -b <background = 0.0>.')
            sys.exit()
        elif opt in ("--id",):
            inputdata = arg
        elif opt in ("--il",):
            inputlabel = arg
        elif opt in ("--od",):
            outputdata = arg
        elif opt in ("--ol",):
            outputlabel = arg
        elif opt in ("-m", '--margin'):
            margin = tuple(int(s) for s in arg.split(','))
            if len(margin) != 3:
                print('Error: The length of margin need to be 3. Example: -m 20,20,20')
                sys.exit() 
        elif opt in ('-b', '--background'):
            background = float(arg)        
        else:
            print('usage: crop_by_label.py --id <input_data_file> --il <input_label_file> --od <output_data_file> --ol <output_data_file> -m <margin = 20,20,20>  -b <background = 0.0>.')
            sys.exit()
    if os.path.isdir(inputdata):
        input_data_files = sorted(glob.glob(os.path.join(inputdata, '*.nii.gz')))
        # then inputlabel should also be a dir
        input_label_files = sorted(glob.glob(os.path.join(inputlabel, '*.nii.gz')))
        for idx in range(len(input_data_files)):
            idfile = input_data_files[idx]
            _, filename = os.path.split(idfile)
            filename, ext = os.path.splitext(filename)
            filename, ext0 = os.path.splitext(filename)
            if ext0 != '':
                ext = ext0 + ext
            odfile = outputdata+'/'+ filename + '_cropped' + ext 

            ilfile = input_label_files[idx]
            _, filename = os.path.split(ilfile)
            filename, ext = os.path.splitext(filename)
            filename, ext0 = os.path.splitext(filename)
            if ext0 != '':
                ext = ext0 + ext
            olfile = outputlabel+'/'+ filename + '_cropped' + ext 
            f(idfile, ilfile, odfile, olfile, margin, background)
    else:
        f(inputdata, inputlabel, outputdata, outputlabel, margin, background)
if __name__ == "__main__":
    main(sys.argv[1:])