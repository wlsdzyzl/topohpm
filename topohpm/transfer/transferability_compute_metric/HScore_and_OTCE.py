from logging import raiseExceptions
from random import sample, shuffle
from token import PERCENT
import numpy as np
import ot
import geomloss
import torch
import math
import os 
from optparse import OptionParser
import json
import torch.nn.functional as F
import time




def compute_ce(P, Y_src, Y_tar):

    src_label_set = set(sorted(list(Y_src.flatten())))
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    # print (src_label_set, tar_label_set)
    P_src_tar = np.zeros((int(np.max(Y_src))+1,int(np.max(Y_tar))+1))
    # print (P_src_tar.shape)

    for y1 in src_label_set:
        y1_idx = np.where(Y_src==y1)
        for y2 in tar_label_set:
            y2_idx = np.where(Y_tar==y2)

            RR = y1_idx[0].repeat(y2_idx[0].shape[0]).astype(np.int)
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0]).astype(np.int)
            P_src_tar[int(y1),int(y2)] = np.sum(P[RR,CC])

    P_src = np.sum(P_src_tar,axis=1)

    entropy = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[int(y1)]
        for y2 in tar_label_set:
            
            if P_src_tar[int(y1),int(y2)] != 0:
                entropy += -(P_src_tar[int(y1),int(y2)] * math.log(P_src_tar[int(y1),int(y2)] / P_y1))

    return entropy 



def compute_coupling(X_src, X_tar):
    
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    C = cost_function(X_src,X_tar)
    C = np.array(C.numpy())
    P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C, numItermax=3000000)
    W = np.sum(P*C)

    return P,W



def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov
    

# compute H-score
def getDiffNN(f,Z, rcond=1e-9):
    #Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    alphabetZ=list(set(Z.reshape((-1,))))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[np.reshape(Z==z, (-1,))], axis=0)
        g[np.reshape(Z==z, (-1,))]=Ef_z
    
    Covg=getCov(g)

    if len(alphabetZ) == 1:
        Covg = np.eye(Covg.shape[0]) / (19 * 19)

    # print ("inverse covf", np.linalg.pinv(Covf,rcond=rcond))
    # print ("covg", Covg)
    dif=np.trace(np.dot(np.linalg.pinv(Covf,rcond=rcond), Covg))
    
    return dif


        

def LEEP(X_tar, Y_tar,num_category=19, h_idx=None, w_idx=None):
    
    NUM_CATEGORY_Z = num_category

    Y_label_set = set(sorted(list(Y_tar.flatten())))
    #num_category_Y = len(Y_label_set)
    num_category_Y = np.max(Y_tar) + 1
    P_yz = np.zeros((num_category_Y, NUM_CATEGORY_Z))
    P_z = np.zeros((NUM_CATEGORY_Z,1))
    num_samples = X_tar.shape[0]
    for i in range(0,num_samples):
          
        P_z[:,0] += X_tar[i]
        P_yz[Y_tar[i],:] += X_tar[i]

    P_z /= num_samples
    P_yz /= num_samples

    P_y_given_z = np.zeros((num_category_Y,NUM_CATEGORY_Z))
    for i in range(0,NUM_CATEGORY_Z):
        
        # avoid NaN results
        if P_z[i] == 0:
            continue
        P_y_given_z[:,i] = P_yz[:,i] / P_z[i]

    leep_score = 0.0
    for i in range(0, num_samples):
        yi = Y_tar[i]
        xi = X_tar[i]
        p_sum = 0.0
        for j in range(0,NUM_CATEGORY_Z):
            p_sum += (P_y_given_z[yi,j] * xi[j])

        leep_score += math.log(p_sum)

    leep_score /= num_samples

    # save_probability_matrix(P_z, P_yz, P_y_given_z, task_dir="./trf_maps/probability_matrix/segnet_gta5_2021-08-30-to-aachen", file_name = "%s_%s_%.4f.npy"%(h_idx, w_idx, leep_score))

    return leep_score



def load_npy_data(feature_map_dir, label_map_dir, num_sample=None, edge_type=None, img_path_list=None, dataset=None):

    rdm = np.random.RandomState(2021)
    file_list = os.listdir(feature_map_dir)

    if num_sample is None:
        NUM_SAMPLES = len(file_list)
        selected_samples = file_list
    else:
        NUM_SAMPLES = num_sample
        selected_samples = rdm.choice(file_list, NUM_SAMPLES)

    start = time.time()
    x = []
    y = []

    for i,file in enumerate(selected_samples):
        map = np.load(os.path.join(feature_map_dir, file)).transpose(1,2,0)
        H,W,C = map.shape
        label = np.load(os.path.join(label_map_dir, file))
        x.append(map)
        y.append(label)

    x = np.array(x)
    y = np.array(y)
    
    return x,y





