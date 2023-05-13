
from skimage.morphology import skeletonize, binary_dilation, binary_erosion
import numpy as np
import torch
import torch.nn as nn
from toposeg.unet3d.losses import DiceLoss

class SkeletonLoss(nn.Module):
    def __init__(self, filter_p = 0.5, loss = 'BCELoss', reduction = 'sum', normalization = 'Sigmoid', dilation_iter = 0, **kwargs):
        super(SkeletonLoss, self).__init__()
        self.filter_p = filter_p
        assert loss in ['MSELoss', 'BCELoss', 'DiceLoss'], "Only MSELoss and BCELoss are supported."
        if loss == 'MSELoss':
            self.loss = nn.MSELoss(reduction = reduction)
        elif loss == 'BCELoss':
            self.loss = nn.BCELoss(reduction = reduction)
        elif loss == 'DiceLoss':
            self.loss = DiceLoss(normalization = 'none')
        if normalization == 'Sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'Softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            print('No normalization layer')
        self.reduction = reduction
        self.dilation_iter = dilation_iter
    def forward(self, pred, target, weight=None):
        if self.normalization is not None:
            # transfer to probability
            pred = self.normalization(pred)
        if weight is not None and weight.shape == pred.shape:
            pred = pred * weight
            target = target * weight
        target = target.type(pred.dtype)
        pred = pred.squeeze(0).squeeze(0)
        target = target.squeeze(0).squeeze(0)
        # add a filter to reduce the number of pd points that we want to optimize
        if pred.shape[0] == 1:
            pred = pred.squeeze(0)
            target = target.squeeze(0)
        gt_binary = target.detach().cpu().numpy() > 0.5
        pred_np = pred.detach().cpu().numpy()
        pred_binary = pred_np > self.filter_p
        ###################### possible improvement
        # gt_dilated = gt_binary.copy()
        # pred_dilated = pred_binary.copy()
        
        # for _ in range(self.dilation_iter):
        #     gt_dilated = binary_dilation(gt_dilated)
        #     pred_dilated = binary_dilation(pred_dilated)
            
        gt_sk = skeletonize(gt_binary)
        pred_sk = skeletonize(pred_binary)
        # pred_sk = skeletonize(pred_np > 1 - self.filter_p)

        positive_selector = np.logical_xor(np.logical_and(gt_sk, pred_binary), gt_sk)
        negative_selector = np.logical_xor(np.logical_and(pred_sk, gt_binary), pred_sk)
        final_selector = np.logical_or(positive_selector, negative_selector)

        ##################### possible improvement
        # if self.dilation_iter > 0:
        #     # during dilation process, some disconnected parts might be connected.
        #     # we need to detect these parts and the skeletons of these parts will also be regarded as critical points
        #     gt_recovered = gt_dilated.copy()
        #     pred_recovered = pred_dilated.copy()
        #     for _ in range(self.dilation_iter):
        #         gt_recovered = binary_erosion(gt_recovered)
        #         pred_recovered = binary_erosion(pred_recovered)
        #     gt_filled = np.logical_xor(gt_recovered, gt_binary)
        #     pred_filled = np.logical_xor(pred_recovered, pred_binary)
        #     filled_selector = np.logical_or(skeletonize(gt_filled), skeletonize(pred_filled))
        #     final_selector = np.logical_or(filled_selector, final_selector)
            
        c_pred = pred[final_selector]
        c_target = target[final_selector]
        loss = torch.Tensor([0.0]).type(pred.dtype).to(pred.device)
        if len(c_pred) > 0:
            loss += self.loss(c_pred, c_target)
        return loss