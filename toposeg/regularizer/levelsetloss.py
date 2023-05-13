from topologylayer.nn import LevelSetLayer2D
from topologylayer.nn.features import *
from topologylayer.util.process import remove_infinite_bars
from topologylayer.nn.levelset import *
import torch, numpy as np
import torch.nn as nn
from toposeg.unet3d.losses import DiceLoss
def remove_nan_bars(dgm):
    sum_dgm = dgm[:,0] + dgm[:, 1]
    return dgm[torch.logical_not(torch.isnan(sum_dgm))]
def remove_short_bars(dgm, min_length = 1e-6):
    sub_dgm = torch.abs(dgm[:, 0] - dgm[:, 1])
    return dgm[sub_dgm > min_length]


def init_grid_3d(x_len, y_len, z_len, maxdim = 2):
    s = SimplicialComplex()
    step_x = y_len * z_len
    step_y = z_len
    step_z = 1
    # 0-cells
    for i in range(x_len):
        for j in range(y_len):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                s.append([ind])
    # 1-cells
    for i in range(x_len-1):
        for j in range(y_len):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                s.append([ind, ind + step_x])
    for i in range(x_len):
        for j in range(y_len - 1):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                s.append([ind, ind + step_y])
    for i in range(x_len):
        for j in range(y_len):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                s.append([ind, ind + 1])
    # 1-cell diagonal and 2-cell
    for i in range(x_len - 1):
        for j in range(y_len - 1):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_x + step_y])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind, ind + step_x, ind + step_x + step_y])
                    s.append([ind, ind + step_y, ind + step_x + step_y])
    # 1-cell diagonal and 2-cell
    for i in range(x_len - 1):
        for j in range(y_len):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_x + step_z])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind, ind + step_x, ind + step_x + step_z])
                    s.append([ind, ind + step_z, ind + step_x + step_z])
    # 1-cell diagonal and 2-cell
    for i in range(x_len):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_y + step_z])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind, ind + step_y, ind + step_y + step_z])
                    s.append([ind, ind + step_z, ind + step_z + step_z])
    if maxdim >= 1:
        for i in range(x_len-1):
            for j in range(y_len - 1):
                for k in range(z_len - 1):
                    ind = i * step_x + j * step_y + k
                    s.append([ind + step_x + step_z, ind + step_y + step_z,  ind])
                    s.append([ind + step_x + step_z, ind + step_y + step_z, ind + step_x + step_y])
                    s.append([ind + step_x, ind + step_y, ind + step_z])
                    s.append([ind + step_x, ind + step_y, ind + step_x + step_y + step_z])
                    s.append([ind + step_x, ind + step_z, ind + step_x + step_y + step_z])
                    s.append([ind + step_y, ind + step_z, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_x + step_z, ind + step_x + step_y])
                    s.append([ind, ind + step_y + step_z, ind + step_x + step_y])
    # 1-cell diagonal, 2-cell, and 3-cell 
    for i in range(x_len-1):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_x + step_y + step_z])
                # 2-cell (6)
                if maxdim >= 1:
                    s.append([ind, ind + step_x, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_y, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_z, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_x + step_y, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_x + step_z, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_y + step_z, ind + step_x + step_y + step_z])
                # 3-cell (12)
                if maxdim >= 2:
                    s.append([ind + step_x, ind + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x, ind + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_y, ind + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x + step_z, ind + step_x + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x + step_y, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x + step_z, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x, ind + step_x + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_y, ind + step_x + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_z, ind + step_x + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x, ind + step_x + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_z, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_y, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])

    # inverse direction
    # 1-cell diagonal and 2-cell
    for i in range(x_len - 1):
        for j in range(y_len - 1):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind + step_x, ind + step_y])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind + step_x, ind + step_y, ind + step_x + step_y])
                    s.append([ind + step_x, ind + step_y, ind])
    # 1-cell diagonal and 2-cell
    for i in range(x_len - 1):
        for j in range(y_len):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind + step_x, ind + step_z])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind + step_x, ind + step_z, ind + step_x + step_z])
                    s.append([ind + step_x, ind + step_z, ind])
    # 1-cell diagonal and 2-cell
    for i in range(x_len):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind + step_y, ind + step_z])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind + step_y, ind + step_z, ind + step_y + step_z])
                    s.append([ind + step_y, ind + step_z, ind])

    # 1-cell diagonal, 2-cell, and 3-cell (not implemented yet)
    for i in range(x_len-1):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind + step_x , ind + step_y + step_z])
                # 2-cell (6)
                if maxdim >= 1:
                    s.append([ind + step_x, ind, ind + step_y + step_z])
                    s.append([ind + step_x, ind + step_y, ind + step_y + step_z])
                    s.append([ind + step_x, ind + step_z, ind + step_y + step_z])
                    s.append([ind + step_x, ind + step_x + step_y, ind + step_y + step_z])
                    s.append([ind + step_x, ind + step_x + step_z, ind + step_y + step_z])
                    s.append([ind + step_x, ind + step_x + step_y + step_z, ind + step_y + step_z])
                # 3-cell (12)
                if maxdim >= 2:
                    s.append([ ind, ind + step_z,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind, ind + step_y,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_y, ind + step_x + step_y,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_z. ind + step_x + step_z,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_x + step_z, ind + step_x + step_y + step_z,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_x + step_y, ind + step_x + step_y + step_z,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_z, ind + step_y,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_z, ind + step_x + step_y + step_z,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_y, ind + step_x + step_y + step_z,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind, ind + step_x + step_z,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind, ind + step_x + step_y,  ind + step_x, ind + step_y + step_z])
                    s.append([ ind + step_x + step_z, ind + step_x + step_y,  ind + step_x, ind + step_y + step_z])

    # 1-cell diagonal, 2-cell, and 3-cell (not implemented yet)
    for i in range(x_len-1):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind + step_y, ind + step_x + step_z])
                # 2-cell (6)
                if maxdim >= 1:
                    s.append([ind + step_y, ind, ind + step_x + step_z])
                    s.append([ind + step_y, ind + step_x, ind + step_x + step_z])
                    s.append([ind + step_y, ind + step_z, ind + step_x + step_z])
                    s.append([ind + step_y, ind + step_x + step_y, ind + step_x + step_z])
                    s.append([ind + step_y, ind + step_y + step_z, ind + step_x + step_z])
                    s.append([ind + step_y, ind + step_x + step_y + step_z, ind + step_x + step_z])
                # 3-cell (12)
                if maxdim >= 2:
                    s.append([ ind, ind + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind, ind + step_x,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_x, ind + step_x + step_y,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_z, ind + step_y + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_x + step_y, ind + step_x + step_y + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_y + step_z, ind + step_x + step_y + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind, ind + step_x + step_y,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind, ind + step_y + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_x + step_y, ind + step_y + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_x, ind + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_z, ind + step_x + step_y + step_z,  ind + step_y, ind + step_x + step_z])
                    s.append([ ind + step_x, ind + step_x + step_y + step_z,  ind + step_y, ind + step_x + step_z])

    # 1-cell diagonal, 2-cell, and 3-cell (not implemented yet)
    for i in range(x_len-1):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind + step_z, ind + step_x + step_y])
                # 2-cell (6)
                if maxdim >= 1:
                    s.append([ind + step_z, ind, ind + step_x + step_y])
                    s.append([ind + step_z, ind + step_x, ind + step_x + step_y])
                    s.append([ind + step_z, ind + step_y, ind + step_x + step_y])
                    s.append([ind + step_z, ind + step_x + step_z, ind + step_x + step_y])
                    s.append([ind + step_z, ind + step_y + step_z, ind + step_x + step_y])
                    s.append([ind + step_z, ind + step_x + step_y + step_z, ind + step_x + step_y])
                # 3-cell (12)
                if maxdim >= 2:
                    s.append([ ind, ind + step_x,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind, ind + step_y,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_x, ind + step_x + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_y, ind + step_y + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_x + step_z, ind + step_x + step_y + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_y + step_z, ind + step_x + step_y + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind, ind + step_x + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind, ind + step_y + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_x + step_z, ind + step_y + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_x, ind + step_y,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_x, ind + step_x + step_y + step_z,  ind + step_z, ind + step_x + step_y])
                    s.append([ ind + step_y, ind + step_x + step_y + step_z,  ind + step_z, ind + step_x + step_y])
    return s


def init_freudenthal_3d(x_len, y_len, z_len, maxdim = 2):
    s = SimplicialComplex()
    step_x = y_len * z_len
    step_y = z_len
    step_z = 1
    # 0-cells
    for i in range(x_len):
        for j in range(y_len):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                s.append([ind])
    # 1-cells
    for i in range(x_len-1):
        for j in range(y_len):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                s.append([ind, ind + step_x])
    for i in range(x_len):
        for j in range(y_len - 1):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                s.append([ind, ind + step_y])
    for i in range(x_len):
        for j in range(y_len):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                s.append([ind, ind + 1])
    # 1-cell diagonal and 2-cell
    for i in range(x_len - 1):
        for j in range(y_len - 1):
            for k in range(z_len):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_x + step_y])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind, ind + step_x, ind + step_x + step_y])
                    s.append([ind, ind + step_y, ind + step_x + step_y])
    # 1-cell diagonal and 2-cell
    for i in range(x_len - 1):
        for j in range(y_len):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_x + step_z])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind, ind + step_x, ind + step_x + step_z])
                    s.append([ind, ind + step_z, ind + step_x + step_z])
    # 1-cell diagonal and 2-cell
    for i in range(x_len):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_y + step_z])
                # 2-cell
                if maxdim >= 1:
                    s.append([ind, ind + step_y, ind + step_y + step_z])
                    s.append([ind, ind + step_z, ind + step_y + step_z])


    # 1-cell diagonal, 2-cell, and 3-cell (not implemented yet)
    for i in range(x_len-1):
        for j in range(y_len - 1):
            for k in range(z_len - 1):
                ind = i * step_x + j * step_y + k
                # 1-cell
                s.append([ind, ind + step_x + step_y + step_z])
                # 2-cell (6)
                if maxdim >= 1:
                    s.append([ind, ind + step_x, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_y, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_z, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_x + step_y, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_x + step_z, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_y + step_z, ind + step_x + step_y + step_z])
                    s.append([ind, ind + step_x + step_z, ind + step_x + step_y])
                    s.append([ind, ind + step_y + step_z, ind + step_x + step_y])
                    s.append([ind + step_x + step_z, ind + step_y + step_z, ind])
                # 3-cell (12)
                if maxdim >= 2:
                    s.append([ind + step_x, ind + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x, ind + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_y, ind + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x + step_z, ind + step_x + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x + step_y, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x + step_z, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x, ind + step_x + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_y, ind + step_x + step_y, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_z, ind + step_x + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_x, ind + step_x + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_z, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])
                    s.append([ind + step_y, ind + step_y + step_z, ind,  ind + step_x + step_y + step_z])
    return s
class LevelSetLayer3D(LevelSetLayer):
    """
    Level set persistence layer for 2D input
    Parameters:
        size : (width, height) - tuple for image input dimensions
        maxdim : maximum homology dimension (default 1)
        sublevel : sub or superlevel persistence (default=True)
        complex : method of constructing complex
            "freudenthal" (default) - canonical triangulation of the lattice
            "grid" - includes diagonals and anti-diagonals
            "delaunay" - scipy delaunay triangulation of the lattice.
                Every square will be triangulated, but the diagonal orientation may not be consistent.
        alg : algorithm
            'hom' = homology (default)
            'cohom' = cohomology
    """
    def __init__(self, size, maxdim=1, sublevel=True, complex="freudenthal", alg='hom'):
        x_len, y_len, z_len = size
        tmpcomplex = None
        if complex == "freudenthal":
            tmpcomplex = init_freudenthal_3d(x_len, y_len, z_len)
        elif complex == "grid":
            tmpcomplex = init_grid_3d(x_len, y_len, z_len)
        super(LevelSetLayer3D, self).__init__(tmpcomplex, maxdim=maxdim, sublevel=sublevel, alg=alg)
        self.size = size




class LabelGuidedBarcodeFeature(nn.Module):

    def __init__(self, dim, min_length = 1e-5, label = 1, remove_zero=True, loss = 'BCELoss', reduction = 'sum'):
        super(LabelGuidedBarcodeFeature, self).__init__()
        self.dim = dim
        self.label = label
        self.remove_zero = remove_zero
        self.min_length = min_length
        assert loss in ['MSELoss', 'BCELoss', 'DiceLoss'], "Only MSELoss and BCELoss are supported."
        if loss == 'MSELoss':
            self.loss = nn.MSELoss(reduction = reduction)
        elif loss == 'BCELoss':
            self.loss = nn.BCELoss(reduction = reduction)
        elif loss == 'DiceLoss':
            self.loss = DiceLoss(normalization = 'none')
    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        dgm = dgms[self.dim]
        dgm = remove_nan_bars(dgm)
        dgm = remove_infinite_bars(dgm, False)
        dgm = remove_short_bars(dgm, self.min_length)
        # print(dgm)
        # log loss or MSE?
        # return torch.sum(torch.pow(1 + self.label - dgm[dgm > 1.0], 2)) + torch.sum(torch.pow(dgm[dgm < 1.0], 2)) 
        target = (dgm > 1.0).type(dgm.dtype)
        dgm[dgm > 1.0] = dgm[dgm > 1.0] - self.label
        filter_idx = target != dgm
        if len(dgm) == 0 or len(dgm[filter_idx]) == 0:
            return torch.Tensor([0.0]).type(dgm.dtype).to(dgm.device)
        return self.loss(dgm[filter_idx], target[filter_idx])


class LevelSetLoss(nn.Module):
    def __init__(self, dims, loss = 'BCELoss', remove_zero=True, normalization = 'Sigmoid', complex_type = 'LevelSet3D', reduction = 'sum', size = None, max_pool = None, maxdim = 0, label = 1, filter_p = 0.6, min_length = 1e-5, sublevel = False, **kwargs):
        super(LevelSetLoss, self).__init__()
        self.complex_layer = None
        # 0 for point cloud and 1 for 3d grids
        self.max_pool = max_pool
        self.normalization = None
        self.dim_of_volume = 3
        self.filter_p = filter_p
        self.min_length = min_length
        if normalization == 'Sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'Softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            print('No normalization layer')
        if not isinstance(dims, list):
            dims = [dims]
        if maxdim < np.max(dims):
            maxdim = np.max(dims)
        if complex_type == "LevelSet3D":
            self.complex_layer = LevelSetLayer3D(size = size, maxdim = maxdim, sublevel = sublevel)
        elif complex_type == "LevelSet2D":
            self.complex_layer = LevelSetLayer2D(size = tuple(reversed(size))[:2], maxdim = maxdim, sublevel = sublevel)      
            self.dim_of_volume = 2
        self.barcode_features = []

        for dim in dims:
            self.barcode_features.append(LabelGuidedBarcodeFeature(dim = dim, loss = loss, label = label, remove_zero = remove_zero, min_length = self.min_length, reduction = reduction))

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
        if self.dim_of_volume == 2:
            pred = pred.squeeze(0)
            target = target.squeeze(0)
        # add a filter to reduce the number of pd points that we want to optimize
        
        final_input = pred + target
        
        filtered_idx = torch.logical_and(pred >= self.filter_p, target > 0.5)
        final_input[filtered_idx] = 2.0

        filtered_idx = torch.logical_and(pred <= (1 - self.filter_p), target < 0.5)
        final_input[filtered_idx] = 0.0

        dgm = self.complex_layer(final_input)
        
        return sum([bf(dgm) for bf in self.barcode_features])
