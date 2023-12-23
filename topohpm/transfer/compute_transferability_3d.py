import numpy as np
import ot
import geomloss
import torch
import math
import os 
from optparse import OptionParser
import json
import torch.nn.functional as F
from collections import Counter
import time

# import LEEP
from topohpm.transfer.transferability_compute_metric.HScore_and_OTCE import LEEP, load_npy_data
# import OTCE 
from topohpm.transfer.transferability_compute_metric.HScore_and_OTCE import compute_coupling, compute_ce
# import LogME
from topohpm.transfer.transferability_compute_metric.LogME import LogME

from topohpm.transfer.transferability_compute_metric.transferability_weighted_map_3d import compute_transferability_weight_map
from topohpm.scripts.utils import load_itk

MAP = []
labels = []
tar_predicted_map_dir = "/home/wlsdzyzl/project/topohpm/generated_files/predictions/3d_asoca/SKELETON_LOSS_0.6_0.5"
tar_gt_map_dir = "/media/wlsdzyzl/DATA2/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label"
# src_feature_map_dir = 'Get_Feature/FeTS2021/source/17_t2/feature/l6'
# src_label_map_dir = 'Get_Feature/FeTS2021/source/17_t2/label'
# src_x, src_y = load_npy_data(src_feature_map_dir, src_label_map_dir, num_sample=200)
'''
print("src_X的shape: ,", src_x.shape)
print("src_y的shape: ,", src_y.shape)
print(type(src_x))
'''

rdm = np.random.RandomState(2021)
tar_file_list = os.listdir(tar_predicted_map_dir)
# selected_samples = rdm.choice(tar_file_list, 200)
# for tar_file in selected_samples:
for tar_file in tar_file_list:
    # print(tar_file)
    map = load_itk(os.path.join(tar_predicted_map_dir, tar_file))[0]
    map = np.stack(( 1 - map, map), axis=-1)
    Z,H,W,C = map.shape
    # map = np.load(os.path.join(tar_predicted_map_dir, tar_file))
    label = load_itk(os.path.join(tar_gt_map_dir, tar_file))[0].astype(int)
    MAP.append(map)
    labels.append(label)
X = np.array(MAP)
y = np.array(labels)


print("X的shape: ,", X.shape)
print("y的shape: ,", y.shape)
print(type(X))

trf_cfg = {'metric':'LEEP', 'num_category':2, 'stride':4 } 
# weight_map = compute_transferability_weight_map(trf_cfg, X, y, src_x, src_y)
weight_map = compute_transferability_weight_map(trf_cfg, X, y)
print("weight map: ", weight_map)
print("weight_map shape:",weight_map.shape)
np.save(os.path.join("resources/3DUnet_asoca_tl/weight_map", 'imagecas_to_asoca_leep.npy'), weight_map)
    
    