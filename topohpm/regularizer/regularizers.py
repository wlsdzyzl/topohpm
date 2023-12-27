from topohpm.scripts.utils import downsample_by_maxpool
from .levelsetloss import LevelSetLoss
from .cubicalloss import CubicalLoss
from .skeletonloss import SkeletonLoss
from .naiveloss import NaiveLoss
from .topoloss import TopoLoss
import numpy as np
import torch
import torch.nn as nn




def compute_levelset_size(original_shape, target_shape):
    # a dirty way to compute the downsampled size
    tmp_array = np.ones(original_shape)
    darray, max_pool = downsample_by_maxpool(tmp_array, target_shape = target_shape)
    return darray.shape, max_pool

def get_regularizer(config):
    assert 'regularizer' in config, 'Could not find regularizer configuration'
    regularizer_config = config['regularizer']
    name = regularizer_config.pop('name','SkeletonLoss')
    if name == 'SkeletonLoss':
        return SkeletonLoss(**regularizer_config)
    elif name == "LevelSetLoss":
        size = None
        max_pool = None
        if 'slice_builder' in config['loaders']['train']:
            patch_shape = config['loaders']['train']['slice_builder']['patch_shape']
        elif 'cropped_shape' in config['loaders']['train']:
            patch_shape = config['loaders']['train']['cropped_shape']
        target_shape = patch_shape
        if 'target_downsampled_shape' in config['trainer']:
            target_shape = config['trainer']['target_downsampled_shape']
        if patch_shape == target_shape:
            size = tuple(patch_shape)
        else:
            size, max_pool = compute_levelset_size(patch_shape, target_shape)
        return TopoRegLevelset(size = size, max_pool = max_pool, **regularizer_config)
    elif name == "CubicalLoss":
        return CubicalLoss(**regularizer_config)
    elif name == "NaiveLoss":
        return NaiveLoss(**regularizer_config)
    elif name == "TopoLoss":
        return TopoLoss(**regularizer_config)
    elif name == "none":
        print('without regularizer')
        return None
    else:
        print('unknow loss function for skeleton loss')
        return None