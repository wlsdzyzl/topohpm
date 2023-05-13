# this file is used to compute prediction error.
# Betti number error (0, 1)
# adapted rand index
# dice coefficient
# accuracy
# Variation of Information(VOI)
# Street Mover distance

from toposeg.unet3d.metrics import MeanIoU, DiceCoefficient
from toposeg.regularized_unet3d.metrics import LevelSetBettiError, CubicalBettiError, Accuracy, VariationOfInformation, AdaptedRandError, AdjustedRandIndex, ConnectedComponentError, CLDice, StreetMoverDistance
import torch, numpy as np
import torch.nn as nn
from collections.abc import Iterable
import sys, getopt
import os
import glob
from toposeg.utils import zoom, load_itk
from skimage.morphology import remove_small_holes, remove_small_objects
import scipy.ndimage as ndimage

def f(pred_path, label_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size=None, prob_threshold = 0.5, area_threshold = 50):
    # print(pred_path, label_path)
    
    pred, _, _ = load_itk(pred_path)
    # print(pred.shape, np.max(pred), np.min(pred))
    pred = pred.astype(float)
    if zoom_size is not None:
        pred = zoom(pred, target_shape = zoom_size)
    # pred /= 255.0

    label, _, _ = load_itk(label_path)
    label = label.astype(float)
    if pred.shape != label.shape:
        label = zoom(label, target_shape = pred.shape)
    # print(label)
    # print(label.shape, np.max(label), np.min(label))
    label[ label >= 0.5 ] = 1.0
    label[ label < 0.5 ] = 0.0

    # remove noisy and disconnected parts
    pred = remove_small_objects(pred > prob_threshold, min_size = area_threshold, connectivity = 3).astype(float)
    label = remove_small_objects(label > prob_threshold, min_size = area_threshold, connectivity = 3).astype(float)
    dice_c = 2 * (pred * label).sum() / (pred.sum() + label.sum()) 
    mIoU_c =  (pred * label).sum() / (pred.sum() + label.sum() - (pred * label).sum())
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
    return res + [dice_c, mIoU_c]

def main(argv):
    pred_path = ''
    gt_path = ''
    threshold = 0.5
    opts, args = getopt.getopt(argv, "hp:g:t:s:z:a:", ['help', 'pred=', 'gt=', 'threshold=', 'size=', 'zoom=', 'area='])
    size = (50, 100, 100)
    zoom_size = None
    area = 50
    if len(opts) == 0:
        print('unknow options, usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -s <size = 50,100,100> -z <zoom = None> -a <area = 50>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -s <size = 50,100,100> -z <zoom = None> -a <area = 50>')
            sys.exit()
        elif opt in ("-p", '--pred'):
            pred_path = arg
        elif opt in ("-g", '--gt'):
            gt_path = arg
        elif opt in ("-t", '--threshold'):
            threshold = float(arg)
        elif opt in ("-s", '--size'):
            size = tuple([int(s) for s in arg.split(',')])
        elif opt in ('-z', '--zoom'):
            zoom_size = tuple([int(z) for z in arg.split(',')])
        elif opt in ('-a', '--area'):
            area = int(arg)
        else:
            print('unknow option,usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5>, -s <size = 50,100,100> -z <zoom = None> -a <area = 50>')
            sys.exit()
    # construct evaluation metrics
    dice = DiceCoefficient(normalization = 'none')
    # cldice = CLDice(threshold = threshold)
    ari = AdjustedRandIndex(threshold = threshold)
    mIoU = MeanIoU(threshold = threshold)
    acc = Accuracy(threshold = threshold)
    vio = VariationOfInformation(threshold = threshold)
    # smd = StreetMoverDistance(threshold = threshold)
    # betti = LevelSetBettiError(size, maxdim = 1, threshold = threshold, rand_patch_number = 1, area_threshold = 0)    
    # betti = CubicalBettiError(size, maxdim=1, threshold = threshold, rand_patch_number = 1)
    betti = ConnectedComponentError(threshold = threshold, dim=3)
    are = AdaptedRandError(threshold = threshold)
    eval_criterions_pixel = [dice, acc, mIoU, betti]
    eval_criterions_cluster = [vio, are]

    if os.path.isdir(pred_path):
        pred_paths = sorted(glob.glob(os.path.join(pred_path, '*')))
        gt_paths = sorted(glob.glob(os.path.join(gt_path, '*')))
        res = []
        for pfile, gfile in zip(pred_paths, gt_paths):
            tmp_res = f(pfile, gfile, eval_criterions_pixel, eval_criterions_cluster, zoom_size, area_threshold = area)
            res.append(tmp_res)
        mean_res = np.mean( np.array(res), axis = 0 )
        print('{:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}'.format(mean_res[0], mean_res[1], mean_res[4], mean_res[5], mean_res[2], mean_res[3], mean_res[6], mean_res[7]))
    else:
        res = f(pred_path, gt_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size, area_threshold = area)
        print(res)
if __name__ == "__main__":
    main(sys.argv[1:])