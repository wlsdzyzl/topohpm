import sys, getopt
import os
import random
import shutil
import math
## tmp file
def main(argv):
    raw_dir = ''
    label_dir = ''
    output_dir = './'
    n = 100
    opts, args = getopt.getopt(argv, "hr:l:o:n:", ['help', 'raw_dir', 'label_dir', 'output_dir', 'num'])
    if len(opts) == 0:
        print('unknow options, usage: batch_process.py -r <raw_dir> -l <label_dir> -o <output_dir=./> -n <num=100>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: batch_process.py -r <raw_dir> -l <label_dir> -o <output_dir=./> -n <num=100>')
            sys.exit()
        elif opt in ("-r", '--raw_dir'):
            raw_dir = arg
        elif opt in ("-l", '--label_dir'):
            label_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('-n', '--num'):
            k = int(arg)
        else:
            print('unknow options, usage: batch_process.py -r <raw_dir> -l <label_dir> -o <output_dir=./> -n <num=100>')
            sys.exit()


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    data = [(os.path.join(raw_dir, raw_image), os.path.join(label_dir, label_image))
            for raw_image, label_image in zip(sorted(os.listdir(raw_dir)), sorted(os.listdir(label_dir)))]
    # Shuffle the data
    random.shuffle(data)
    data = data[:n]
    #### more reasonable
    # fold_size = math.floor(len(data) / k)
    # folds = [data[i:i+fold_size] for i in range(0, len(data), fold_size)]
    # if len(folds) > k:
    #     final_fold = folds[-1]
    #     folds = folds[:k]
    #     for i, f in enumerate(final_fold):
    #         folds[i].append(f)


    fold_raw_dir = os.path.join(output_dir, f"raw")
    fold_label_dir = os.path.join(output_dir, f"label")
    if os.path.isdir(fold_raw_dir):
        shutil.rmtree(fold_raw_dir)
    if os.path.isdir(fold_label_dir):
        shutil.rmtree(fold_label_dir)
    os.makedirs(fold_raw_dir)
    os.makedirs(fold_label_dir)

    print(f'creating fold ...') 
    for raw_image_path, label_image_path in data:
        print(f'copy {raw_image_path} to {fold_raw_dir} and {label_image_path} to {fold_label_dir}'  ) 
        shutil.copy(raw_image_path, os.path.join(fold_raw_dir, os.path.basename(raw_image_path)))
        shutil.copy(label_image_path, os.path.join(fold_label_dir, os.path.basename(label_image_path)))
if __name__ == "__main__":
    main(sys.argv[1:])