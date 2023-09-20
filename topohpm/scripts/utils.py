import numpy as np
import SimpleITK as sitk
import scipy
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData,PlyElement
import mcubes
import nibabel as nb

def load_itk(filename):
    # Reads the image using SimpleITK
    try:
        itkimage = sitk.ReadImage(filename)
    except:
        print('Orthonormal direction error occurs, try to fix it')
        img = nb.load(filename)
        qform = img.get_qform()
        sform = img.get_sform()
        img.set_qform(qform)
        img.set_sform(sform)
        nb.save(img, filename)
        itkimage = sitk.ReadImage(filename)
        print('Done.')
    imageArray = sitk.GetArrayFromImage(itkimage)
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()

    return imageArray, origin, spacing

def save_itk(filename, imageArray, origin = None, spacing = None):
    itkimage = sitk.GetImageFromArray(imageArray)
    if origin is not None:
        itkimage.SetOrigin(origin)
    if spacing is not None:
        itkimage.SetSpacing(spacing)
    sitk.WriteImage(itkimage, filename, useCompression = True)

def read_sdf(mhd_path):
    volumeArray, volumeOrigin, volumeSpacing = load_itk(mhd_path+".mhd")
    with open(mhd_path+".sdf", 'r') as f:
        line = f.readline().strip('\n')
        line = line.strip(' ')
        sdf_str_array = line.split(' ')
        print(len(sdf_str_array))
        sdf_array = np.array([float(x) for x in sdf_str_array])
        outputVolumeArray = np.reshape(sdf_array, volumeArray.shape, order='F')
    return (outputVolumeArray, volumeOrigin, volumeSpacing)

# from mhd to nii.gz
def mhd2nii(mhd_file, nii_file):
    data, origin, spacing = load_itk(mhd_file)
    save_itk(nii_file, data, origin = origin, spacing = spacing)
def nii2mhd(nii_file, mhd_file):
    data, origin, spacing = load_itk(nii_file)
    save_itk(mhd_file, data, origin = origin, spacing = spacing)

def npy2nii(npy_file, nii_file):
    data = np.load(npy_file)
    save_itk(nii_file, data)
def nii2npy(nii_file, npy_file):
    data, _, _ = load_itk(nii_file)
    np.save(npy_file, img)

def nrrd2nii(nrrd_file, nii_file):
    data, origin, spacing = load_itk(nrrd_file)
    save_itk(nii_file, data, origin = origin, spacing = spacing)
def nii2nrrd(nii_file, nrrd_file):
    data, origin, spacing= load_itk(nii_file)
    save_itk(nrrd_file, data, origin = origin, spacing = spacing)


def get_coordinates(volume_size, dtype = float):
    dimension = len(volume_size)
    return np.stack(np.meshgrid(*[np.arange(0.0, size).astype(dtype) for size in volume_size], indexing='ij'), axis=-1).reshape(-1, dimension)
# '_t' indicates the function works with tensor, not nparray
def get_coordinates_t(volume_size, dtype = torch.FloatTensor):
    dimension = len(volume_size)
    return torch.stack(torch.meshgrid(*[torch.arange(0.0, size).type(dtype) for size in volume_size], indexing='ij'), dim = dimension).view(-1, dimension)


def write_tensor(filename, skeleton):
    X = skeleton.data.cpu().numpy()
    np.savetxt(filename, X)
def write_sphere_t(filename, skeleton, radius, local_coor):
    s = skeleton.data.cpu().numpy()
    r = radius.data.cpu().numpy()
    lc = local_coor.data.cpu().numpy()
    X = np.repeat(s, lc.shape[0], axis = 0) + np.tile(lc, (s.shape[0], 1)) * np.repeat(r, lc.shape[0])[:, None] 
    np.savetxt(filename, X)

def zoom(X, scaling = (0.5, 0.5, 0.5), target_shape = None):
    mode = 'bilinear'
    if len(X.shape) == 3:
        mode = 'trilinear'
    # why do we use torch.interpolate instead of zoom in scipy
    if target_shape is not None:
        output_volume = F.interpolate(torch.from_numpy(X).unsqueeze(0).unsqueeze(0), size = target_shape, mode = mode, recompute_scale_factor = False).squeeze().squeeze().cpu().detach().numpy()
    else:
        output_volume = F.interpolate(torch.from_numpy(X).unsqueeze(0).unsqueeze(0), scale_factor = scaling, mode = mode, recompute_scale_factor = False).squeeze().squeeze().cpu().detach().numpy()
    return output_volume
def downsample_by_maxpool(X, scaling = (0.5, 0.5, 0.5), target_shape = None):
    if target_shape is not None:
        scaling = tuple(np.array(target_shape).astype(float) / np.array(X.shape).astype(float))
    kernel_size = tuple( int(1 / s) for s in scaling)
    max_pool = nn.MaxPool3d(kernel_size)
    output_volume = max_pool(torch.from_numpy(X).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).cpu().detach().numpy()
    return output_volume, max_pool
def downsample_by_maxpool_t(X, scaling = (0.5, 0.5, 0.5), target_shape = None):
    if target_shape is not None:
        scaling = tuple(np.array(target_shape).astype(float) / np.array(X.shape).astype(float)[-3:])
    kernel_size = tuple( int(1 / s) for s in scaling)

    max_pool = nn.MaxPool3d(kernel_size)
    if len(X.shape) == 3:
        X = X.unsqueeze(0).unsqueeze(0)
    output_volume = max_pool(X).squeeze(0).squeeze(0)
    return output_volume, max_pool
def crop_middle(X, scaling = (0.5, 0.5, 0.5), target_shape = None):
    data_shape = np.array(X.shape)
    if target_shape is None:
        scaling = np.array(scaling)
        target_shape = (data_shape * scaling).astype(int)
    else:
        target_shape = np.array(target_shape).astype(int)
    res = np.zeros(target_shape)
    # data start idx
    start_idx = ((data_shape - target_shape) / 2).astype(int)
    start_idx[start_idx < 0] = 0

    end_idx = start_idx + target_shape
    end_idx[end_idx > data_shape] = data_shape[end_idx > data_shape]

    # res start idx
    valid_shape = end_idx - start_idx
    res_start_idx = ((target_shape - valid_shape)/ 2).astype(int)
    res_start_idx[res_start_idx < 0] = 0
    res_end_idx = res_start_idx + valid_shape
    res[res_start_idx[0]:res_end_idx[0], res_start_idx[1]:res_end_idx[1], res_start_idx[2]:res_end_idx[2]] = X[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
    return res, start_idx

def get_boundingbox(X, background = 0):
    coor = get_coordinates(X.shape)
    selector = (X.reshape(-1) != background)
    valid_points = coor[selector]
    return np.min(valid_points, axis=0), np.max(valid_points, axis=0)
# X is label, Y is data
def crop_by_boundingbox(label, data = None, margin = (10, 10, 10), background = 0):
    start_idx, end_idx = get_boundingbox(label)
    print(start_idx, end_idx)
    start_idx = start_idx - margin
    start_idx[start_idx < 0] = 0
    end_idx = end_idx + margin
    end_idx = np.minimum(end_idx, np.array(label.shape))
    start_idx = start_idx.astype(int)
    end_idx = end_idx.astype(int)
    cropped_label = label[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
    if data is not None:
        cropped_data = data[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
        return cropped_label, cropped_data, start_idx
    return cropped_label, start_idx
# mask to point cloud
def write_ply(filename, points, pcolors = None):
    if pcolors is not None:
        if np.max(pcolors) > 1 or np.min(pcolors) < 0:
            pcolor = (pcolors - np.min(pcolors))/ (np.max(pcolors) - np.min(pcolors))
        pcolors = (pcolors * 255).astype(int)
        if len(pcolors.shape) < 2 or pcolors.shape[1] < 3:
            pcolors = pcolors.reshape((-1))
            points = [(points[i, 0], points[i, 1], points[i, 2], pcolors[i], pcolors[i], pcolors[i]) for i in range(points.shape[0])]
        else:
            points = [(points[i, 0], points[i, 1], points[i, 2], pcolors[i, 0], pcolors[i, 2], pcolors[i, 2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1') ])
    else:
        points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(vertex, 'vertex', comments = ['vertices'])
    PlyData([el], text=True).write(filename)
# mask to mesh
def write_obj(filename, mask, smooth = True):
    if smooth:
        mask = mcubes.smooth(mask)
        mask = mask + 0.5
    vertices, triangles = mcubes.marching_cubes(mask, 0.5)
    mcubes.export_obj(vertices, triangles, filename)