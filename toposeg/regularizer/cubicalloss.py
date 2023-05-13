import torch, numpy as np
import torch.nn as nn
import gudhi as gd
from toposeg.unet3d.losses import DiceLoss
### GUDHI version topological regularizer
def get_pd_ids(pred, dims = [1]):
    """
    Compute the critical points of the image (Value range from 0 -> 1)
    Args:
        pred: pred image from the output of the neural networks
    Returns:
        pd_pred:  persistence diagram.
        bcp_pred: Birth critical points.
        dcp_pred: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.
    """
    dimension = pred.ndim

    pred_vector = np.asarray(1 - pred).flatten()
    if dimension == 2:
        pred_cubic = gd.CubicalComplex(
            dimensions=[pred.shape[0], pred.shape[1]],
            top_dimensional_cells=pred_vector
        )
    elif dimension == 3:
        pred_cubic = gd.CubicalComplex(
            dimensions=[pred.shape[0], pred.shape[1], pred.shape[2]],
            top_dimensional_cells=pred_vector
        )        
    pred_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    pairs_pred = pred_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_pred[0])==0): return [], []
    res_pd = []
    res_pd_idx = []
    for dim in dims:
        if len(pairs_pred[0]) <= dim:
            continue
        bd_pairs = pairs_pred[0][dim]
        if dimension == 2:
            birth_id = np.hstack( ((bd_pairs[:, 0] // pred.shape[1])[:, None], (bd_pairs[:, 0] % pred.shape[1])[:, None] ))
            death_id = np.hstack( ((bd_pairs[:, 1] // pred.shape[1])[:, None], (bd_pairs[:, 1] % pred.shape[1])[:, None] ))
        elif dimension == 3:
            birth_id = np.hstack( ( (bd_pairs[:, 0] // (pred.shape[1] * pred.shape[2]))[:, None], (bd_pairs[:, 0] % (pred.shape[1] * pred.shape[2]) // pred.shape[2]) [:, None], (bd_pairs[:, 0] % (pred.shape[1] * pred.shape[2]) % pred.shape[2]) [:, None]) )
            death_id = np.hstack( ( (bd_pairs[:, 1] // (pred.shape[1] * pred.shape[2]))[:, None], (bd_pairs[:, 1] % (pred.shape[1] * pred.shape[2]) // pred.shape[2]) [:, None], (bd_pairs[:, 1] % (pred.shape[1] * pred.shape[2]) % pred.shape[2]) [:, None]) )
        # get persistent diagram
        current_pd = np.hstack((pred_vector[bd_pairs[:, 0]][:, None], pred_vector[bd_pairs[:, 1]][:, None] ))
        current_pd_len = current_pd[:, 1] - current_pd[:, 0]
        # [::-1], reverse the vector    get sorted_idx
        sorted_idx = np.argsort(current_pd_len)[::-1]
        birth_id = birth_id[sorted_idx]
        death_id = death_id[sorted_idx]
        current_pd = current_pd[sorted_idx]
        res_pd.append(current_pd)
        res_pd_idx.append((birth_id, death_id))
    return res_pd, res_pd_idx

class CubicalLoss(nn.Module):
    def __init__(self, dims, loss = 'BCELoss', remove_zero=True, reduction = 'mean', normalization = 'Sigmoid', max_pool = None, filter_p = 0.6, **kwargs):
        super(CubicalLoss, self).__init__()
        # 0 for point cloud and 1 for 3d grids
        self.max_pool = max_pool
        self.normalization = None
        self.filter_p = filter_p
        if normalization == 'Sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'Softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            print('No normalization layer')
        self.dims = dims
        if not isinstance(dims, list):
            self.dims = [dims]

        assert loss in ['MSELoss', 'BCELoss', 'DiceLoss'], "Only MSELoss and BCELoss are supported."
        if loss == 'MSELoss':
            self.loss = nn.MSELoss(reduction = reduction)
        elif loss == 'BCELoss':
            self.loss = nn.BCELoss(reduction = reduction)
        elif loss == 'DiceLoss':
            self.loss = DiceLoss(normalization = 'none')
    def forward(self, pred, target, weight = None):
        if self.normalization is not None:
            # transfer to probability
            pred = self.normalization(pred)
        target = target.type(pred.dtype)
        if self.max_pool is not None:
            pred = self.max_pool(pred)
            target = self.max_pool(target)
        if weight is not None and weight.shape == pred.shape:
            pred = pred * weight
            target = target * weight
        pred = pred.squeeze(0).squeeze(0)
        target = target.squeeze(0).squeeze(0)
        # add a filter to reduce the number of pd points that we want to optimize
        if pred.shape[0] == 1:
            pred = pred.squeeze(0)
            target = target.squeeze(0)
        final_input = pred + 0.0
        
        filtered_idx = torch.logical_and(pred >= self.filter_p, target > 0.5)
        final_input[filtered_idx] = 1.0
        # filtered_idx = torch.logical_and(pred <= 1 - self.filter_p, target < 0.5)
        filtered_idx = torch.logical_and(pred <= self.filter_p, target < 0.5)
        final_input[filtered_idx] = 0.0

        dgms, dgm_ids = get_pd_ids(final_input.cpu().detach().numpy(), self.dims)
        loss = torch.Tensor([0.0]).type(pred.dtype).to(pred.device)
        for dim_idx in range(len(dgms)):
            dgm = dgms[dim_idx]
            birth_idx, death_idx = dgm_ids[dim_idx]
            # do we need to remove zero bars? non bars? infinite bars?
            overall_idx = np.vstack((birth_idx, death_idx))
            if overall_idx.shape[1] == 2:
                dim_input = final_input[overall_idx[:, 0], overall_idx[:, 1]]
                dim_target = target[overall_idx[:, 0], overall_idx[:, 1]]
            elif overall_idx.shape[1] == 3:
                dim_input = final_input[overall_idx[:, 0], overall_idx[:, 1], overall_idx[:, 2]]
                dim_target = target[overall_idx[:, 0], overall_idx[:, 1], overall_idx[:, 2]]                
            selector = dim_input != dim_target
            if len(dim_input[selector]) != 0:
                loss = loss + self.loss( dim_input[selector], dim_target[selector] )
        return loss