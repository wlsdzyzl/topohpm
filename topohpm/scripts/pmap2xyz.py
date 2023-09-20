from utils import *
import sys, getopt
import os
import glob
def f(inputfile, outputfile, prob_threshold):
    img_array, _, _ = load_itk(inputfile)
    selector = (img_array > prob_threshold).reshape((-1))
    locations = (get_coordinates(img_array.shape))[selector]
    print('extract {0:d} valid points, save to {1:s}'.format(len(locations), outputfile))
    np.savetxt(outputfile, locations)
def main(argv):
    inputfile = ''
    outputfile = ''
    prob_threshold = 0.5
    opts, args = getopt.getopt(argv, "hi:o:p:", ['help', 'input=', 'output=', 'prob_threshold='])
    if len(opts) == 0:
        print('unknow options, usage: pmap2xyz.py -i <inputfile> -o <outputfile> -p <prob_threshold = 0.5>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: pmap2xyz.py -i <inputfile> -o <outputfile> -p <prob_threshold = 0.5>')
            sys.exit()
        elif opt in ("-i", '--input'):
            inputfile = arg
        elif opt in ("-o", '--output'):
            outputfile = arg
        elif opt in ("-p", '--prob_threshold'):
            prob_threshold = float(arg)
        else:
            print('unknow option, usage: pmap2xyz.py -i <inputfile> -o <outputfile> -p <prob_threshold = 0.5>')
            sys.exit()

    if os.path.isdir(inputfile):
        input_files = sorted(glob.glob(os.path.join(inputfile, '*.nii.gz')))
        for ifile in input_files:
            _, filename = os.path.split(ifile)
            filename, _ = os.path.splitext(filename)
            filename, _ = os.path.splitext(filename)
            ofile = outputfile+'/'+ filename + '.xyz'
            f(ifile, ofile, prob_threshold)
    else:
        f(inputfile, outputfile, prob_threshold)
if __name__ == "__main__":
    main(sys.argv[1:])