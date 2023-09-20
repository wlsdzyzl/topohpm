import sys, getopt
import os
import random
import shutil
import math
import imageio
import numpy as np
## tmp file
def main(argv):
    raw_dir = ''
    label_dir = ''
    output_dir = './'
    k = 5
    opts, args = getopt.getopt(argv, "hr:l:o:k:", ['help', 'raw_dir', 'label_dir', 'output_dir', 'kfold'])
    if len(opts) == 0:
        print('unknow options, usage: batch_process.py -r <raw_dir> -l <label_dir> -o <output_dir=./> -k <kfold=5>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: batch_process.py -r <raw_dir> -l <label_dir> -o <output_dir=./> -k <kfold=5>')
            sys.exit()
        elif opt in ("-r", '--raw_dir'):
            raw_dir = arg
        elif opt in ("-l", '--label_dir'):
            label_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('-k', '--kfold'):
            k = int(arg)
        else:
            print('unknow options, usage: batch_process.py -r <raw_dir> -l <label_dir> -o <output_dir=./> -k <kfold=5>')
            sys.exit()


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    data = [(os.path.join(raw_dir, raw_image), os.path.join(label_dir, label_image))
            for raw_image, label_image in zip(sorted(os.listdir(raw_dir)), sorted(os.listdir(label_dir)))]
    fold_raw_dir = os.path.join(output_dir, "raw")
    fold_label_dir = os.path.join(output_dir, "label")
    if os.path.isdir(fold_raw_dir):
        shutil.rmtree(fold_raw_dir)
    if os.path.isdir(fold_label_dir):
        shutil.rmtree(fold_label_dir)
    os.makedirs(fold_raw_dir)
    os.makedirs(fold_label_dir)
    # Shuffle the data
    for raw_path, label_path in data:
        img = np.asarray(imageio.imread(raw_path))
        # detect invalid sample
        img = np.sum(img, axis=2)
        print(np.sum(img >= 255 * 3) / (1500 * 1500))
        if np.sum(img >= 255 * 3) / (1500 * 1500) < 0.1:  
            print(f'copy {raw_path} to {fold_raw_dir} and {label_path} to {fold_label_dir}'  ) 
            shutil.copy(raw_path, os.path.join(fold_raw_dir, os.path.basename(raw_path)))
            shutil.copy(label_path, os.path.join(fold_label_dir, os.path.basename(label_path)))
if __name__ == "__main__":
    main(sys.argv[1:])