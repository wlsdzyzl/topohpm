import torch, numpy as np
import torch.nn as nn
import imageio

import sys, getopt
import os
import glob

from topologylayer.nn import LevelSetLayer2D
from topologylayer.nn.features import *
from topologylayer.util.process import remove_zero_bars, remove_infinite_bars
from toposeg.regularizer.levelsetloss import remove_nan_bars, remove_short_bars
import gudhi as gd

from skimage.morphology import skeletonize

def detect_critical_points_naive(pred, gt):
    positive_idx = np.logical_and(pred < 0.55, label > 0.5)
    negative_idx = np.logical_and(pred > 0.45, label < 0.5)    
    return positive_idx, negative_idx
def detect_critical_points_skeleton(pred, gt):
    gt_binary = gt > 0.5
    pred_binary = pred > 0.5
    gt_sk = skeletonize(gt_binary)
    pred_sk = skeletonize(pred_binary)

    positive = np.logical_xor(np.logical_and(gt_sk, pred_binary), gt_sk)
    negative = np.logical_xor(np.logical_or(pred_sk, gt_binary), gt_binary)

    return positive, negative

def detect_critical_points_gudhi(likelihood, label, dims = [1]):
    """
    Compute the critical points of the image (Value range from 0 -> 1)
    Args:
        likelihood: Likelihood image from the output of the neural networks
    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.
    """
    likelihood[ np.logical_and(likelihood > 0.75, label > 0.5 )] = 1.0
    likelihood[ np.logical_and(likelihood < 0.25, label < 0.5 )] = 0.0

    lh = 1 - likelihood
    lh_vector = np.asarray(lh).flatten()
    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return [], []
    print(pairs_lh)
    # return persistence diagram, birth/death critical points
    positive_idx = np.full(likelihood.shape, False)
    negative_idx = np.full(likelihood.shape, False)
    for dim in dims:
        bd_pairs = pairs_lh[0][dim]
        birth_id = np.hstack( ((bd_pairs[:, 0] // likelihood.shape[1])[:, None], (bd_pairs[:, 0] % likelihood.shape[1])[:, None] ))
        death_id = np.hstack( ((bd_pairs[:, 1] // likelihood.shape[1])[:, None], (bd_pairs[:, 1] % likelihood.shape[1])[:, None] ))
        overall_id = np.vstack((birth_id, death_id)).astype(int)
        # print(label.shape)
        positive_selector = np.logical_and(label[overall_id[:, 0], overall_id[:, 1]] > 0.5, likelihood[overall_id[:, 0], overall_id[:, 1]] < 1.0)
        negative_selector = np.logical_and(label[overall_id[:, 0], overall_id[:, 1]] < 0.5, likelihood[overall_id[:, 0], overall_id[:, 1]] > 0.0)
        
        tmp_positive_idx = overall_id[positive_selector]
        tmp_negative_idx = overall_id[negative_selector]
        positive_idx[tmp_positive_idx[:, 0], tmp_positive_idx[:, 1]] = True
        negative_idx[tmp_negative_idx[:, 0], tmp_negative_idx[:, 1]] = True
    return positive_idx, negative_idx

def detect_critical_points(pred, label, complex_layer, dims = [1]):
    positive_idx = torch.full(pred.shape, False)
    negative_idx = torch.full(pred.shape, False)
    # filter
    # print(np.random.normal(size = pred.shape) / 1e10)
    pred = pred + np.random.normal(size = pred.shape) / 1e10
    pred = torch.from_numpy(pred).float()
    label = torch.from_numpy(label).float()

    pred[ torch.logical_and(pred > 0.55, label > 0.5 )] = 1.0
    pred[ torch.logical_and(pred < 0.45, label < 0.5 )] = 0.0
    final_map = pred + label
    imageio.imwrite('tmp.png', final_map.cpu().numpy())

    dgms, issublevel = complex_layer(final_map)
    for dim in dims:
        dgm = dgms[dim]
        dgm = remove_nan_bars(dgm)
        dgm = remove_infinite_bars(dgm, False)
        dgm = remove_short_bars(dgm, 1.0)    
        target = (dgm > 1.0).float()
        target = target + target
        filter_idx = (dgm != target)
        dgm = dgm[filter_idx]
        target = target[filter_idx]
        print(len(dgm))
        for i in range(len(dgm)):
            print(dgm[i])
            print(torch.sum(final_map == dgm[i]))
            if target[i] > 0.5:
                positive_idx = torch.logical_or(positive_idx, final_map == dgm[i])
            else:
                negative_idx = torch.logical_or(negative_idx, final_map == dgm[i])

    # positive_idx = torch.logical_and(pred < 0.55, label > 0.5)
    # negative_idx = torch.logical_and(pred > 0.45, label < 0.5)
    print(torch.sum(positive_idx) + torch.sum(negative_idx))
    return positive_idx.cpu().numpy(), negative_idx.cpu().numpy()

tmp = imageio.imread("/media/*****/DATA/datasets/DRIVE/training/images/21_training.tif")
print(tmp.shape)
pred_path = "/home/*****/project/toposeg/generated_files/predictions/2d/CREMI/unet/test/prob/1.png"
label_path = "/media/*****/DATA/datasets/CREMI/dataset_A/test/label/1.png"
pred = np.asarray(imageio.imread(pred_path)).astype(float)
pred /= 255.0
label = np.asarray(imageio.imread(label_path)).astype(float)
label[ label > 0 ] = 1.0

complex_layer = LevelSetLayer2D(size = (pred.shape[1], pred.shape[0]), maxdim = 1, sublevel = False)

res = np.zeros((pred.shape[0], pred.shape[1], 3))
res[:, :, 0] = pred
res[:, :, 1] = pred
res[:, :, 2] = pred
# positive_idx, negative_idx = detect_critical_points_skeleton(pred, label)
positive_idx, negative_idx = detect_critical_points_gudhi(pred, label)
# positive_idx, negative_idx = detect_critical_points_naive(pred, label)

res[:, :, 0][negative_idx] = 1.0
res[:, :, 1][negative_idx] = 0.0
res[:, :, 2][negative_idx] = 0.0

res[:, :, 0][positive_idx] = 0.0
res[:, :, 1][positive_idx] = 0.0
res[:, :, 2][positive_idx] = 1.0

imageio.imwrite('critical_points.png', res)