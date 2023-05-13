# from hd5 to png

import h5py
import numpy as np
import os
import imageio
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation
import scipy
def create_border_mask_2d(image, max_dist):
    """
    Create binary border mask for image.
    A pixel is part of a border if one of its 4-neighbors has different label.
    
    Parameters
    ----------
    image : numpy.ndarray - Image containing integer labels.
    max_dist : int or float - Maximum distance from border for pixels to be included into the mask.
    Returns
    -------
    mask : numpy.ndarray - Binary mask of border pixels. Same shape as image.
    """
    max_dist = max(max_dist, 0)
    
    padded = np.pad(image, 1, mode='edge')
    
    border_pixels = np.logical_and(
        np.logical_and( image == padded[:-2, 1:-1], image == padded[2:, 1:-1] ),
        np.logical_and( image == padded[1:-1, :-2], image == padded[1:-1, 2:] )
        )

    distances = scipy.ndimage.distance_transform_edt(
        border_pixels,
        return_distances=True,
        return_indices=False
        )

    return distances <= max_dist

def f(input_h5py, raw_internal_path, neuronids_internal_path, output_dir):
    input_file = h5py.File(input_h5py, 'r')
    # print(list(input_file.keys()))
    output_raw_path = output_dir + '/raw'
    output_boundary_path = output_dir + '/label'

    if not os.path.isdir(output_raw_path):
        os.makedirs(output_raw_path)
    if not os.path.isdir(output_boundary_path):
        os.makedirs(output_boundary_path)
    raw_data = input_file[raw_internal_path][:]
    nid_data = input_file[neuronids_internal_path][:]
    print("write images to", output_raw_path, output_boundary_path)
    for idx in range(len(raw_data)):
        # boundary = find_boundaries(nid_data[idx], mode='thick', connectivity = 2)
        # boundary = binary_dilation(boundary)
        # boundary = binary_dilation(boundary).astype('uint8') * 255
        boundary = create_border_mask_2d(nid_data[idx], 2).astype('uint8') * 255
        imageio.imwrite(output_raw_path + '/' + str(idx) +'.png', raw_data[idx])
        imageio.imwrite(output_boundary_path + '/' + str(idx) +'.png', boundary)
        


f('/media/*****/DATA/datasets/CREMI/sample_A_20160501.hdf', 'volumes/raw', 'volumes/labels/neuron_ids', '/media/*****/DATA/datasets/CREMI/dataset_A_cremi')
# f('/media/*****/DATA/datasets/CREMI/sample_B_20160501.hdf', 'volumes/raw', 'volumes/labels/neuron_ids', '/media/*****/DATA/datasets/CREMI/dataset_B')
# f('/media/*****/DATA/datasets/CREMI/sample_C_20160501.hdf', 'volumes/raw', 'volumes/labels/neuron_ids', '/media/*****/DATA/datasets/CREMI/dataset_C')