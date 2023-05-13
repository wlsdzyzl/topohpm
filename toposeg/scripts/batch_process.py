from utils import *
import sys, getopt
import os
import glob
## tmp file
def f(inputfile, outputfile):
    # img_array, origin, spacing = load_itk(inputfile)
    # # img_array = 1 - img_array #(img_array > 0).astype(float)
    # img_array[img_array > 0.5] = 1.0
    # img_array[img_array <= 0.5] = 0.0
    # save_itk(outputfile, img_array, origin = origin, spacing = spacing)
    nrrd2nii(inputfile, outputfile)
def main(argv):
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input', 'output'])
    if len(opts) == 0:
        print('unknow options, usage: batch_process.py -i <inputfile> -o <outputfile>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: batch_process.py -i <inputfile> -o <outputfile> ')
            sys.exit()
        elif opt in ("-i", '--input'):
            inputfile = arg
        elif opt in ("-o", '--output'):
            outputfile = arg
        else:
            print('unknow option, usage: batch_process.py -i <inputfile> -o <outputfile>')
            sys.exit()

    if os.path.isdir(inputfile):
        if not os.path.isdir(outputfile):
            os.makedirs(outputfile)
        input_files = sorted(glob.glob(os.path.join(inputfile, '*.nrrd')))
        for ifile in input_files:
            _, filename = os.path.split(ifile)
            filename, _ = os.path.splitext(filename)
            filename, _ = os.path.splitext(filename)
            ofile = outputfile+'/'+ filename + '.nii.gz'
            f(ifile, ofile)
    else:
        f(inputfile, outputfile)
if __name__ == "__main__":
    main(sys.argv[1:])