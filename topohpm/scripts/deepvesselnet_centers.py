from utils import *
import sys, getopt
import os
import glob
def f(inputfile_center, inputfile_radius, outputfile, prob_threshold):
    img_array_c, _, _ = load_itk(inputfile_center)
    img_array_r, _, _ = load_itk(inputfile_radius)
    selector = (img_array_c > prob_threshold).reshape((-1))
    radius = img_array_r.reshape((-1))[selector]
    
    locations = (get_coordinates(img_array_c.shape))[selector]
    # locations += 0.5
    xyzr = np.hstack((locations, radius[:, None]))
    print('extract {0:d} valid points, save to {1:s}'.format(len(xyzr), outputfile))
    np.savetxt(outputfile, xyzr)
    np.savetxt(outputfile[:-1], locations)
def main(argv):
    inputfile_center = ''
    inputfile_radius = ''
    outputfile = ''
    prob_threshold = 0.5
    opts, args = getopt.getopt(argv, "hc:r:o:p:", ['help', 'center=', 'radois=', 'output=', 'prob_threshold='])
    if len(opts) == 0:
        print('unknow options, usage: pmap2xyz.py -c <centerline> -r <radius> -o <outputfile> -p <prob_threshold = 0.5>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: pmap2xyz.py -c <centerline> -r <radius> -o <outputfile> -p <prob_threshold = 0.5>')
            sys.exit()
        elif opt in ("-c", '--center'):
            inputfile_center = arg
        elif opt in ("-r", '--radius'):
            inputfile_radius = arg
        elif opt in ("-o", '--output'):
            outputfile = arg
        elif opt in ("-p", '--prob_threshold'):
            prob_threshold = float(arg)
        else:
            print('unknow option, usage: pmap2xyz.py -c <centerline> -r <radius> -o <outputfile> -p <prob_threshold = 0.5>')
            sys.exit()

    if os.path.isdir(inputfile_center):
        input_files_c = sorted(glob.glob(os.path.join(inputfile_center, '*.nii.gz')))
        input_files_r = sorted(glob.glob(os.path.join(inputfile_radius, '*.nii.gz')))
        for ifilec, ifiler in zip(input_files_c, input_files_r):
            _, filename = os.path.split(ifilec)
            filename, _ = os.path.splitext(filename)
            filename, _ = os.path.splitext(filename)
            ofile = outputfile+'/'+ filename + '.xyzr'
            f(ifilec, ifiler, ofile, prob_threshold)
    else:
        f(inputfile_center, inputfile_radius, outputfile, prob_threshold)
if __name__ == "__main__":
    main(sys.argv[1:])