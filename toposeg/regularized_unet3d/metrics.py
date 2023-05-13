from topologylayer.nn import LevelSetLayer2D
from topologylayer.nn.features import *
from topologylayer.util.process import remove_zero_bars, remove_infinite_bars
import torch, numpy as np
import torch.nn as nn
from skimage.metrics import variation_of_information as voi, adapted_rand_error as are
from sklearn.metrics.cluster import adjusted_rand_score as ari
from skimage.morphology import remove_small_holes, remove_small_objects
from toposeg.regularizer.levelsetloss import remove_nan_bars, LevelSetLayer3D
import gudhi as gd
import scipy.ndimage as ndimage
import imageio
import ot
from skimage.morphology import skeletonize
from toposeg.scripts.utils import get_coordinates
from geomloss import SamplesLoss
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class LevelSetBettiError:
    def __init__(self, size, threshold = 0.5, maxdim = 1, area_threshold = 64, rand_patch_number = 10, min_size = 100,  **kwargs):
        self.threshold = threshold
        self.layer = None
        if len(size) == 2:
            self.layer = LevelSetLayer2D(size = tuple(reversed(size)), maxdim = maxdim, complex="grid", sublevel = False)
        elif len(size) == 3:
            self.layer = LevelSetLayer3D(size = size, maxdim = maxdim, sublevel = False)
        self.maxdim = maxdim
        self.size = size
        self.area_threshold = area_threshold
        self.min_size = min_size
        self.rand_patch_number = rand_patch_number
    def __call__(self, input, target):
        # remove small holes
        input = remove_small_holes(input > self.threshold, area_threshold = self.area_threshold, connectivity = 2)
        target = remove_small_holes(target > self.threshold, area_threshold = self.area_threshold, connectivity = 2)

        input = remove_small_objects(input > self.threshold, min_size = self.area_threshold, connectivity = 2)
        target = remove_small_objects(target > self.threshold, min_size = self.area_threshold, connectivity = 2)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()
        
        res = []
        rand_patch_idx = np.random.randint(low = 0, high = input.shape[0] - self.size[0] + 1, size = (self.rand_patch_number, 1))
        rand_patch_idy = np.random.randint(low = 0, high = input.shape[1] - self.size[1] + 1, size = (self.rand_patch_number, 1))
        rand_patch_ids = np.hstack((rand_patch_idx, rand_patch_idy))
        for pid in rand_patch_ids:
            current_input_patch = input[pid[0]:pid[0] + self.size[0], pid[1]:pid[1] + self.size[1]]
            current_target_patch = target[pid[0]:pid[0] + self.size[0], pid[1]:pid[1] + self.size[1]]
            input_dgms, _ = self.layer(current_input_patch)
            target_dgms, _ = self.layer(current_target_patch)
            tmp_res = []
            for idx in range(self.maxdim+1):
                input_dgm = input_dgms[idx]
                target_dgm = target_dgms[idx]

                input_dgm = remove_nan_bars(input_dgm)
                input_dgm = remove_infinite_bars(input_dgm, False)
                input_dgm = remove_zero_bars(input_dgm)

                target_dgm = remove_nan_bars(target_dgm)
                target_dgm = remove_infinite_bars(target_dgm, False)
                target_dgm = remove_zero_bars(target_dgm)
                tmp_res.append(len(input_dgm) - len(target_dgm))
                print(len(input_dgm), len(target_dgm))
            res.append(tmp_res)
            # print(res)
        return np.mean(np.abs(res), axis = 0)
class CubicalBettiError:
    def __init__(self, size, threshold = 0.5,  maxdim = 1, rand_patch_number = 10, area_threshold = 64, **kwargs):
        self.rand_patch_number = rand_patch_number
        self.size = size
        self.threshold = threshold
        self.maxdim = maxdim
        self.area_threshold = area_threshold
    def __call__(self, input, target):
        assert input.shape == target.shape, "the shapes of input and target should be the same."
        input = remove_small_holes(input > self.threshold, area_threshold = self.area_threshold, connectivity = 2)
        target = remove_small_holes(target > self.threshold, area_threshold = self.area_threshold, connectivity = 2)

        input = remove_small_objects(input > self.threshold, min_size = self.area_threshold, connectivity = 2)
        target = remove_small_objects(target > self.threshold, min_size = self.area_threshold, connectivity = 2)
        res = []
        input[input > self.threshold] = 1.0
        input[input <= self.threshold] = 0.0

        target[target > self.threshold] = 1.0
        target[target <= self.threshold] = 0.0

        rand_patch_idx = np.random.randint(low = 0, high = input.shape[0] - self.size[0] + 1, size = (self.rand_patch_number, 1))
        rand_patch_idy = np.random.randint(low = 0, high = input.shape[1] - self.size[1] + 1, size = (self.rand_patch_number, 1))
        rand_patch_ids = np.hstack((rand_patch_idx, rand_patch_idy))
        dimension = input.ndim
        for pid in rand_patch_ids:
            current_input_patch = input[pid[0]:pid[0] + self.size[0], pid[1]:pid[1] + self.size[1]]
            current_target_patch = target[pid[0]:pid[0] + self.size[0], pid[1]:pid[1] + self.size[1]]

            input_vector = np.asarray(1 - input).flatten()
            if dimension == 2:
                input_cubic = gd.CubicalComplex(
                    dimensions=[input.shape[0], input.shape[1]],
                    top_dimensional_cells=input_vector
                )
            elif dimension == 3:
                input_cubic = gd.CubicalComplex(
                    dimensions=[input.shape[0], input.shape[1], input.shape[2]],
                    top_dimensional_cells=input_vector
                )   
            input_cubic.persistence(homology_coeff_field=2, min_persistence=0)
            pairs_input = input_cubic.cofaces_of_persistence_pairs()

            target_vector = np.asarray(1 - target).flatten()
            if dimension == 2:
                target_cubic = gd.CubicalComplex(
                    dimensions=[target.shape[0], target.shape[1]],
                    top_dimensional_cells=target_vector
                )
            elif dimension == 3:
                target_cubic = gd.CubicalComplex(
                    dimensions=[target.shape[0], target.shape[1], target.shape[2]],
                    top_dimensional_cells=target_vector
                )   
            target_cubic.persistence(homology_coeff_field=2, min_persistence=0)
            pairs_target = target_cubic.cofaces_of_persistence_pairs()


            tmp_res = []
            for dim in range(self.maxdim+1):
                input_bd_pairs = pairs_input[0][dim]
                target_bd_pairs = pairs_target[0][dim]


                input_pd_len = input_vector[input_bd_pairs[:, 1]] - input_vector[input_bd_pairs[:, 0]]
                target_pd_len = target_vector[target_bd_pairs[:, 1] - target_bd_pairs[:, 0]]
                # print(input_pd_len)
                tmp_res.append((input_pd_len > 0).sum() - (target_pd_len > 0).sum())
            res.append(tmp_res)
            # print(res)
        return np.mean(np.abs(res), axis = 0)
## Error of connected components
# full structure
class ConnectedComponentError:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target):


        input = (input > self.threshold).astype(int)
        target = (target > self.threshold).astype(int)
        _, nf_i = ndimage.label(input, structure = self.structure)
        _, nf_t = ndimage.label(target, structure = self.structure)
        b0_error = np.abs(0.0 + nf_i - nf_t)

        input = 1 - input
        target = 1 - target
        _, nf_i = ndimage.label(input, structure = self.structure)
        _, nf_t = ndimage.label(target, structure = self.structure)

        # imageio.imwrite('fuck1.png', input * 255)
        # imageio.imwrite('fuck2.png', target * 255)
        # exit(0)
        b1_error = np.abs(0.0 + nf_i - nf_t)
        return [b0_error, b1_error] 

class Accuracy:
    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
    def __call__(self, input, target):
        input = (input > self.threshold).astype(float)
        target = (target > self.threshold).astype(float)
        res = (input == target).astype(float)
        return np.sum(res) / np.size(res)



# street mover distance (wait to be implemented)
class VariationOfInformation:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target, is_boundary = True):
        if is_boundary:
            input = (input < self.threshold).astype(int)
            target = (target < self.threshold).astype(int)
            input_label, _ = ndimage.label(input, structure = self.structure)
            target_label, _ = ndimage.label(target, structure = self.structure)
            return sum(voi(input_label, target_label)) / 2
        else:
            return sum(voi(input, target)) / 2

class AdjustedRandIndex:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target, is_boundary = True):
        if is_boundary:
            input = (input < self.threshold).astype(int)
            target = (target < self.threshold).astype(int)
            input_label, _ = ndimage.label(input, structure = self.structure)
            target_label, _ = ndimage.label(target, structure = self.structure)
            return ari(input_label.flatten(), target_label.flatten())
        else:
            return ari(input.flatten(), target.flatten())
class AdaptedRandError:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target, is_boundary = True):
        if is_boundary:
            input = (input < self.threshold).astype(int)
            target = (target < self.threshold).astype(int)
            input_label, _ = ndimage.label(input, structure = self.structure)
            target_label, _ = ndimage.label(target, structure = self.structure)
            return are(input_label, target_label)[0]
        else:
            return are(input, target)[0]
def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]
    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]
    """
    tprec = cl_score(v_p,skeletonize(v_l))
    tsens = cl_score(v_l,skeletonize(v_p))
    return tsens, tprec

class CLDice:
    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
    def __call__(self, input, target):
        input = input > self.threshold
        target = target > self.threshold
        return clDice(input, target)
class WassersteinLoss(nn.Module):
    def __init__(self, loss = SamplesLoss("sinkhorn", blur=0.01, scaling=0.9), **kwargs):
        super(WassersteinLoss, self).__init__()
        self.loss = loss
    def forward(self, source_points, target_points, prob_threshold = 0.5):
        # construct sinkhorn distance
        source_weights = torch.full([source_points.shape[0]], 1.0 / (source_points.shape[0])).to(source_points.device).type(torch.FloatTensor)
        target_weights = torch.full([target_points.shape[0]], 1.0 / (target_points.shape[0])).to(target_points.device).type(torch.FloatTensor)
        # from mask volume to target
        return  self.loss(source_weights, source_points, target_weights, target_points)
# wasserstein distance between two point clouds
def w_distance(input, target):
    M = ot.dist(input, target)
    # print(input.shape, target.shape)
    a = torch.from_numpy(np.ones((input.shape[0], )) / input.shape[0]).to(device)
    b = torch.from_numpy(np.ones((target.shape[0], )) / target.shape[0]).to(device)
    M = torch.from_numpy(M / 1000).to(device)
    # print(a.dtype, b.dtype)
    return ot.sinkhorn2(a, b, M, 1)
class StreetMoverDistance:
    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
        self.wloss = WassersteinLoss()
        self.wloss = self.wloss.to(device)
    def __call__(self, input, target):
        input_selector = skeletonize(input > self.threshold).reshape((-1))
        target_selector = skeletonize(target > self.threshold).reshape((-1))
        all_p = get_coordinates(input.shape)
        input_pcd = torch.from_numpy(all_p[input_selector]).to(device).type(torch.FloatTensor)
        target_pcd = torch.from_numpy(all_p[target_selector]).to(device).type(torch.FloatTensor)
        # return w_distance(all_p[input_selector], all_p[target_selector])
        return self.wloss(input_pcd, target_pcd)


