from utils import *
import sys, getopt
from torch.nn import Parameter
import os
import glob
import torch
from topohpm.skeleton.soft_skeleton import soft_skel

# soft skeletonization cannot get the exact skeleton
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float64

def f(inputfile, outputfile, n, scaling):
    max_epoch = 10000000
    max_init_epoch = 200
    # magic number
    alpha = 5
    beta = 1000
    prob_volume, _, _ = load_itk(inputfile)
    
    skeleton_volume = soft_skel(torch.from_numpy(prob_volume).type(dtype).to(device).unsqueeze(0).unsqueeze(0), n)
    skeleton_np = zoom(skeleton_volume.squeeze().squeeze().data.cpu().numpy(), (scaling, scaling, scaling)) 
    selector = skeleton_np.flatten()
    X = get_coordinates(skeleton_np.shape)[selector > 0.5]
    np.savetxt(outputfile, X)
    
def main(argv):
    inputfile = ''
    outputfile = './'
    num_iter = 50
    scaling = 0.75
    opts, args = getopt.getopt(argv, "hi:o:n:s:", ['help', 'input=', 'output=', 'num_iter=', 'scaling='])
    if len(opts) == 0:
        print('unknow options, usage: extract_skeleton_soft.py -i <input_prob_volume> -o <outputfile = "./"> -n <num_iter = 100> -s <scaling = 0.75> ')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: extract_skeleton_soft.py -i <input_prob_volume> -o <outputfile = "./"> -n <num_iter = 100> -s <scaling = 0.75>')
            sys.exit()
        elif opt in ("-i", '--input'):
            inputfile = arg
        elif opt in ("-o", '--output'):
            outputfile = arg
        elif opt in ("-n", '--num_iter'):
            num_iter = int(arg)
        elif opt in ('-s', '--scaling'):
            scaling = float(arg)
        else:
            print('usage: extract_skeleton_soft.py -i <input_prob_volume> -o <outputfile = "./"> -n <num_iter = 100> -s <scaling = 0.75>')
            sys.exit()
    if os.path.isdir(inputfile):
        input_files = sorted(glob.glob(os.path.join(inputfile, '*.nii.gz')))
        for ifile in input_files:
            _, filename = os.path.split(ifile)
            filename, _ = os.path.splitext(filename)
            filename, _ = os.path.splitext(filename)

            ofile = outputfile+'/'+ filename + '.xyz'
            f(ifile, ofile, num_iter, scaling)

    else:
        f(inputfile, outputfile, num_iter, scaling)
if __name__ == '__main__':
    main(sys.argv[1:])