# this file is used to compute prediction error.
# Betti number error (0, 1)
# adapted rand index
# dice coefficientmIoU
# accuracy
# Variation of Information(VOI)
# Street Mover distance

from topohpm.regularized_unet3d.np_metrics import MeanIoU, Dice, Accuracy, VariationOfInformation, AdaptedRandError, ConnectedComponentError
from skimage import metrics
import numpy as np
import imageio
from collections.abc import Iterable
import sys, getopt
import os
import glob
from topohpm.scripts.utils import zoom
from skimage.morphology import remove_small_holes, remove_small_objects
import scipy.ndimage as ndimage

def f(pred_path, label_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size=None, prob_threshold = 0.5, area_threshold = 10, min_size = 100):
    # print(pred_path, label_path)
    
    pred = np.asarray(imageio.imread(pred_path)).astype(float)
    if zoom_size is not None:
        pred = zoom(pred, target_shape = zoom_size)
    if len(pred.shape) == 3:
        pred = np.mean(pred, axis=2)
    pred /= 255.0
    label = np.asarray(imageio.imread(label_path)).astype(float)
    if pred.shape != label.shape:
        label = zoom(label, target_shape = pred.shape)
    # print(label)
    label[ label < 128.0] = 0.0
    label[ label >= 128.0] = 1.0
    
    
    pred = remove_small_holes(pred > prob_threshold, area_threshold = area_threshold, connectivity = 2).astype(float)
    label = remove_small_holes(label > prob_threshold, area_threshold = area_threshold, connectivity = 2).astype(float)
    # remove noisy and disconnected parts
    pred = remove_small_objects(pred > prob_threshold, min_size = min_size, connectivity = 2).astype(float)
    label = remove_small_objects(label > prob_threshold, min_size = min_size, connectivity = 2).astype(float)
    dice_c = 2 * (pred * label).sum() / (pred.sum() + label.sum())     
    res =[dice_c]
    for eval_c in eval_criterions_pixel:
        tmp = eval_c(pred, label)
        if isinstance(tmp, Iterable):
            res = res + list(tmp)
        else:
            res.append(tmp)
    structure = np.ones((3,3), int)
    input_label, _ = ndimage.label(np.logical_not(pred), structure = structure)
    target_label, _ = ndimage.label(np.logical_not(label), structure = structure)  
        
    for eval_c in eval_criterions_cluster:
        tmp = eval_c(input_label, target_label, False)
        if isinstance(tmp, Iterable):
            res = res + list(tmp)
        else:
            res.append(tmp)
    return res
def main(argv):
    pred_path = ''
    gt_path = ''
    threshold = 0.5
    opts, _ = getopt.getopt(argv, "hp:g:t:z:a:m:", ['help', 'pred=', 'gt=', 'threshold=','zoom=', 'area=', 'minsize='])

    area = 10
    min_size = 100
    zoom_size = None
    if len(opts) == 0:
        print('unknow options, usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -z <zoom = None> -a <area = 10> -m <minsize = 100>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -z <zoom = None> -a <area = 10> -m <minsize = 100>')
            sys.exit()
        elif opt in ("-p", '--pred'):
            pred_path = arg
        elif opt in ("-g", '--gt'):
            gt_path = arg
        elif opt in ("-t", '--threshold'):
            threshold = float(arg)
        elif opt in ('-z', '--zoom'):
            zoom_size = tuple([int(z) for z in arg.split(',')])
        elif opt in ("-a", '--area'):
            area = int(arg)
        elif opt in ('-m', '--minsize'):
            min_size = int(arg)
        else:
            print('unknow option,usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5>, -z <zoom = None> -a <area = 10> -m <minsize = 100>')
            sys.exit()

    acc = Accuracy(threshold = threshold)
    vio = VariationOfInformation(threshold = threshold)
    mIoU = MeanIoU(threshold = threshold)
    dice = Dice(threshold=threshold)
    betti = ConnectedComponentError(threshold = threshold, dim=2)
    are = AdaptedRandError(threshold = threshold)
    eval_criterions_pixel = [acc, mIoU, dice, betti]
    eval_criterions_cluster = [vio, are]
    if os.path.isdir(pred_path):
        pred_paths = sorted(glob.glob(os.path.join(pred_path, '*')))
        gt_paths = sorted(glob.glob(os.path.join(gt_path, '*')))
        res = []
        for pfile, gfile in zip(pred_paths, gt_paths):
            # print(pfile, gfile)
            tmp_res = f(pfile, gfile, eval_criterions_pixel, eval_criterions_cluster, zoom_size, prob_threshold = threshold, area_threshold = area, min_size = min_size)
            res.append(tmp_res)
            # print(tmp_res)
        mean_res = np.mean( np.array(res), axis = 0 )
        print('{:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}'.format(mean_res[0], mean_res[1], mean_res[4], mean_res[5], mean_res[2], mean_res[3]))
    else:
        res = f(pred_path, gt_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size, prob_threshold = threshold, area_threshold = area, min_size = min_size)
        print(res)
if __name__ == "__main__":
    main(sys.argv[1:])