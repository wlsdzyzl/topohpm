import torch
import torch.nn.functional as F
import numpy as np


def cross_entropy2d(input, target, weight=None, reduction='mean'):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    # print("n,c,h,w is :", n,c,h,w)
    # print("nt,ht,wt is :", nt,ht,wt)
    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )

    return loss


def pixel_weighted_cross_entropy2d(input, target, weight_map, size_average=False): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # process weightmap
    weight_map = weight_map[np.newaxis,np.newaxis,:,:]
    weight_map = np.repeat(weight_map, nt, axis=0)
    weight_map = torch.tensor(weight_map).to(device)
     
    _, _, hw, ww = weight_map.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt: # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    if hw != ht and ww != wt:
        weight_map = F.interpolate(weight_map, size=(ht, wt), mode="nearest")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    weight_map = weight_map.view(-1)
 
    # 设置reduction为None表示不进行聚合，返回一个loss数组    
    loss = F.cross_entropy(input, target, ignore_index=250, reduction='none')

    loss = loss * weight_map

    return loss.mean()


def pixel_weighted_cross_entropy3d(input, target, weight_map, size_average=False): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    n, c, z, h, w = input.size()
    nt, zt, ht, wt = target.size()

    # process weightmap
    weight_map = weight_map[np.newaxis,np.newaxis,:,:,:]
    weight_map = np.repeat(weight_map, nt, axis=0)
    weight_map = torch.tensor(weight_map).to(device)
     
    _, _, zw, hw, ww = weight_map.size()

    # Handle inconsistent size between input and target
    if z != zt and h != ht and w != wt: # upsample labels
        input = F.interpolate(input, size=(zt, ht, wt), mode="bilinear", align_corners=True)

    if zw != zt and hw != ht and ww != wt:
        weight_map = F.interpolate(weight_map, size=(zt, ht, wt), mode="nearest")

    input = input.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(-1)
    weight_map = weight_map.view(-1)
 
    # 设置reduction为None表示不进行聚合，返回一个loss数组    
    loss = F.cross_entropy(input, target, ignore_index=250, reduction='none')

    loss = loss * weight_map

    return loss.mean()
