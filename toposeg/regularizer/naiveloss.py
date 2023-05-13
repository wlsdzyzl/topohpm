
import numpy as np
import torch
import torch.nn as nn
from toposeg.unet3d.losses import DiceLoss

class NaiveLoss(nn.Module):
    def __init__(self, filter_p = 0.5, loss = 'BCELoss', reduction = 'sum', normalization = 'Sigmoid', **kwargs):
        super(NaiveLoss, self).__init__()
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

        positive_selector = torch.logical_and(pred < self.filter_p, target > 0.5)
        # negative_selector = torch.logical_and(pred > 1 - self.filter_p, target < 0.5)
        negative_selector = torch.logical_and(pred > self.filter_p, target < 0.5)        

        c_pred = pred[torch.logical_or(positive_selector, negative_selector)]
        c_target = target[torch.logical_or(positive_selector, negative_selector)]
        loss = torch.Tensor([0.0]).type(pred.dtype).to(pred.device)
        if len(c_pred) > 0:
            loss += self.loss(c_pred, c_target)
        return loss