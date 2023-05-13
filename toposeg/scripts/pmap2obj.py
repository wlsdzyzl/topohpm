from utils import *
import sys, getopt
import os
import glob
def f(inputfile, outputfile, prob_threshold, smooth):
    img_array, _, _ = load_itk(inputfile)
    print(img_array.shape)
    mask = img_array > prob_threshold
    write_obj(outputfile, mask, smooth)
def main(argv):
    inputfile = ''
    outputfile = ''
    prob_threshold = 0.5
    smooth = False
    opts, args = getopt.getopt(argv, "hi:o:p:s:", ['help', 'input=', 'output=', 'prob_threshold=', 'smooth='])
    if len(opts) == 0:
        print('unknow options, usage: pmap2obj.py -i <inputfile> -o <outputfile> -p <prob_threshold = 0.5> -s <smooth = 0>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: pmap2obj.py -i <inputfile> -o <outputfile> -p <prob_threshold = 0.5> -s <smooth = 0>')
            sys.exit()
        elif opt in ("-i", '--input'):
            inputfile = arg
        elif opt in ("-o", '--output'):
            outputfile = arg
        elif opt in ("-p", '--prob_threshold'):
            prob_threshold = float(arg)
        elif opt in ("-s", '--smooth'):
            smooth = float(arg) > 0
        else:
            print('unknow option, usage: pmap2obj.py -i <inputfile> -o <outputfile> -p <prob_threshold = 0.5> -s <smooth = 0>')
            sys.exit()

    if os.path.isdir(inputfile):
        if not os.path.isdir(outputfile):
            os.makedirs(outputfile)
        input_files = sorted(glob.glob(os.path.join(inputfile, '*.nii.gz')))
        for ifile in input_files:
            _, filename = os.path.split(ifile)
            filename, _ = os.path.splitext(filename)
            filename, _ = os.path.splitext(filename)
            ofile = outputfile+'/'+ filename + '.obj'
            f(ifile, ofile, prob_threshold, smooth)
    else:
        f(inputfile, outputfile, prob_threshold, smooth)
if __name__ == "__main__":
    main(sys.argv[1:])