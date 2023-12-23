import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import yaml
import random
from tqdm import tqdm

# import LEEP
from transferability_compute_metric.HScore_and_OTCE import LEEP, load_npy_data
# import LogME
from transferability_compute_metric.LogME import LogME
# import H-score
from transferability_compute_metric.HScore_and_OTCE import getDiffNN
# import OTCE 
from transferability_compute_metric.HScore_and_OTCE import compute_coupling, compute_ce

import matplotlib.pyplot as plt



def compute_transferability_weight_map(args, tar_x, tar_y, src_x=None, src_y=None):
    
    # device = torch.device("cuda:%d"%(args.gpu_id) if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    rdm = np.random.RandomState(2021)

    total_score = 0.0
    start = time.time()

    N,Z,H,W,C = tar_x.shape
    stride = args['stride']
    trf_map = np.zeros((int(H / stride), int(H / stride), int(W / stride)))
    
    if args['metric'] == 'LEEP':
        tar_x_tensor = torch.from_numpy(tar_x)
        tar_x = F.softmax(tar_x_tensor,dim=-1).numpy()

    # if args['transfer_setting'] == 'ADE20K':
    #     num_category = 19 #150
    # else:
    #     num_category = 19
        
    num_category = args['num_category']
        
    
    for z in range(int(Z / stride)):
        for i in range(int(H / stride)):
            for j in range(int(W / stride)):
                
                cur_patch_x = tar_x[:,z*stride:(z+1)*stride,i*stride:(i+1)*stride,j*stride:(j+1)*stride,:].reshape(-1,C)
                cur_patch_y = tar_y[:,z*stride:(z+1)*stride,i*stride:(i+1)*stride,j*stride:(j+1)*stride].reshape(-1)
                
                # filter 250
                ignore_idx = np.where(cur_patch_y==250)
                cur_patch_y_filted = np.delete(cur_patch_y, ignore_idx)
                cur_patch_x_filted = np.delete(cur_patch_x,ignore_idx[0], axis=0)
                
                if cur_patch_x_filted.shape[0] < 5:
                    continue

                if args['metric'] == 'LEEP':
                    
                    # negative transferability是为了提高可迁移性低的pixel的权重
                    score = - LEEP(cur_patch_x_filted, cur_patch_y_filted,num_category=num_category, h_idx=i, w_idx=j)
                    
                    # if is_weight_map:
                    #     score = np.exp(-score)
                    # else:
                    #     score = np.exp(score)
                    
                elif args['metric'] == 'H-score':
                    score = - getDiffNN(cur_patch_x_filted, cur_patch_y_filted, rcond=1e-4).astype(np.float)
                    # score = score.astype(np.float)
                    # print ("Hscore", -score)
                    # save_path = os.path.join('./trf_maps/%d_%d/'%(i,j))
                    # if not os.path.exists(save_path):
                    #     os.makedirs(save_path)
                    # np.save(os.path.join(save_path,'inv_covf.npy'), inv_covf)
                    # np.save(os.path.join(save_path,'covg.npy'), covg)
                
                elif args['metric'] == 'LogME':
                    logme = LogME(regression=False)
                    score = - logme.fit(cur_patch_x_filted, cur_patch_y_filted)
                
                elif args['metric'] == 'OTCE':
                    

                    _,Hs,Ws,Cs = src_x.shape
                    
                    if Cs != C:
                        raise Exception("src feature and target feature should have the same dimension")

                    H_ratio = Hs / H
                    W_ratio = Ws / W
                    
                    i_s = int(round(i * H_ratio))
                    j_s = int(round(j * W_ratio))
                    
                    src_patch_x = src_x[:,i_s*stride:(i_s+1)*stride,j_s*stride:(j_s+1)*stride,:].reshape(-1,Cs)
                    src_patch_y = src_y[:,i_s*stride:(i_s+1)*stride,j_s*stride:(j_s+1)*stride].reshape(-1)
                    
                    # filter 250
                    src_ignore_idx = np.where(src_patch_y==250)
                    src_patch_y_filted = np.delete(src_patch_y, src_ignore_idx)
                    src_patch_x_filted = np.delete(src_patch_x, src_ignore_idx[0], axis=0)
                    
                    if src_patch_x_filted.shape[0] < 5:
                        continue
                    
                    src_patch_x_filted_tensor = torch.from_numpy(src_patch_x_filted)
                    cur_patch_x_filted_tensor = torch.from_numpy(cur_patch_x_filted)

                    P, W_dis = compute_coupling(src_patch_x_filted_tensor, cur_patch_x_filted_tensor)
                    score = compute_ce(P,src_patch_y_filted, cur_patch_y_filted)
                
                else:
                    raise Exception("no such option")
                
                
                trf_map[z,i,j] = score
    
    
    # normalize 0-1, 并排除掉ignore区域的影响
    if args['metric'] == 'LEEP' or args['metric'] == 'OTCE':
        trf_map_normalize = (trf_map - np.min(trf_map[trf_map>0])) / (np.max(trf_map[trf_map>0]) - np.min(trf_map[trf_map>0]))
    else:
        min_ = np.min(trf_map[trf_map<0])
        max_ = np.max(trf_map[trf_map<0])
        if trf_map[trf_map>0].shape[0] > 0:
            max_ = np.max(trf_map[trf_map>0])

        # print (min_, max_)

        trf_map_normalize = (trf_map - min_) / (max_ - min_)
    
    trf_map_normalize[trf_map==0] = 0
    # exponential
    trf_map_normalize = np.exp(trf_map_normalize)

    # trf_map_normalize = trf_map
    
    end = time.time()
    print ("time: %.2f"%(end-start))
    return trf_map_normalize
