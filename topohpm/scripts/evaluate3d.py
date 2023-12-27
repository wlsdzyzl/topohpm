# this file is used to compute prediction error.
# Betti number error (0, 1)
# adapted rand index
# dice coefficient
# accuracy
# Variation of Information(VOI)
# Street Mover distance

# from topohpm.unet3d.metrics import MeanIoU, DiceCoefficient
from topohpm.regularized_unet3d.np_metrics import  Accuracy, MeanIoU, Dice, VariationOfInformation, AdaptedRandError, ConnectedComponentError
import torch.nn as nn
from collections.abc import Iterable
import sys, getopt
import os
import glob
from topohpm.scripts.utils import zoom, load_itk
from skimage.morphology import remove_small_holes, remove_small_objects
import scipy.ndimage as ndimage
from utils import *
def f(pred_path, label_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size=None, prob_threshold = 0.5, area_threshold = 50):
    # print(pred_path, label_path)
    label, _, _ = load_itk(label_path)
    # print(label_path)
    label = label.astype(float)    

    pred, _, _ = load_itk(pred_path)
    # print(pred.shape, np.max(pred), np.min(pred))
    pred = pred.astype(float)
    
    if zoom_size is not None:
        pred = zoom(pred, target_shape = zoom_size)
    # pred /= 255.0

    # print(pred.shape, label.shape)
    if pred.shape != label.shape:
        label = zoom(label, target_shape = pred.shape)
    # print(label)
    # print(label.shape, np.max(label), np.min(label))
    label[ label >= 0.5 ] = 1.0
    label[ label < 0.5 ] = 0.0
    # remove noisy and disconnected parts
    pred = remove_small_objects(pred > prob_threshold, min_size = area_threshold, connectivity = 3).astype(float)
    label = remove_small_objects(label > prob_threshold, min_size = area_threshold, connectivity = 3).astype(float)
    label = label > prob_threshold
    ## for shape reconstruction from skeletal representation
    # e_label = ndimage.binary_erosion(label)
    # label = e_label.astype(float) +  label.astype(float) - ndimage.binary_dilation(e_label).astype(float)
    res = []
    for eval_c in eval_criterions_pixel:
        tmp = eval_c(pred, label)
        if torch.is_tensor(tmp):
            tmp = tmp.item()
        if isinstance(tmp, Iterable):
            res = res + list(tmp)
        else:
            res.append(tmp)

    structure = np.ones((3,3,3), int)
    # print(pred.shape, structure.shape)
    input_label, _ = ndimage.label(np.logical_not(pred), structure = structure)
    target_label, _ = ndimage.label(np.logical_not(label), structure = structure)  
        
    for eval_c in eval_criterions_cluster:
        tmp = eval_c(input_label, target_label, False)
        if torch.is_tensor(tmp):
            tmp = tmp.item()
        # print(tmp)
        if isinstance(tmp, Iterable):
            res = res + list(tmp)
        else:
            res.append(tmp)
    return res

def main(argv):
    pred_path = ''
    gt_path = ''
    threshold = 0.5
    opts, _ = getopt.getopt(argv, "hp:g:t:z:a:", ['help', 'pred=', 'gt=', 'threshold=', 'zoom=', 'area='])
    zoom_size = None
    area = 50
    if len(opts) == 0:
        print('unknow options, usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -z <zoom = None> -a <area = 50>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -z <zoom = None> -a <area = 50>')
            sys.exit()
        elif opt in ("-p", '--pred'):
            pred_path = arg
        elif opt in ("-g", '--gt'):
            gt_path = arg
        elif opt in ("-t", '--threshold'):
            threshold = float(arg)
        elif opt in ('-z', '--zoom'):
            zoom_size = tuple([int(z) for z in arg.split(',')])
        elif opt in ('-a', '--area'):
            area = int(arg)
        else:
            print('unknow option,usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5>, -z <zoom = None> -a <area = 50>')
            sys.exit()
    # # construct evaluation metrics
    mIoU = MeanIoU(threshold = threshold)
    dice = Dice(threshold=threshold)
    vio = VariationOfInformation(threshold = threshold)
    betti = ConnectedComponentError(threshold = threshold, dim=3)
    are = AdaptedRandError(threshold = threshold)
    eval_criterions_pixel = [mIoU, dice, betti]
    eval_criterions_cluster = [vio, are]

    if os.path.isdir(pred_path):
        pred_paths = sorted(glob.glob(os.path.join(pred_path, '*')))
        gt_paths = sorted(glob.glob(os.path.join(gt_path, '*')))
        res = []
        for pfile, gfile in zip(pred_paths, gt_paths):
            # print(pfile, gfile)
            tmp_res = f(pfile, gfile, eval_criterions_pixel, eval_criterions_cluster, zoom_size, area_threshold = area)
            hasnan = True if True in np.isnan(np.array(tmp_res)) else False
            if not hasnan:
                res.append(tmp_res)
            # print(tmp_res)
        mean_res = np.mean( np.array(res), axis = 0 )
        # print('--------------------------------------------------------------')
        print(mean_res)
    else:
        res = f(pred_path, gt_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size, area_threshold = area)
        print(res)
if __name__ == "__main__":
    main(sys.argv[1:])