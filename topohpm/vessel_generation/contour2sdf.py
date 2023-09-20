import numpy as np
from numpy import linalg as LA
import scipy as cp
from scipy.spatial import cKDTree
from contour_generation import *
import SimpleITK as sitk

def get_coordinates(volume_size, dtype = float):
    dimension = len(volume_size)
    return np.stack(np.meshgrid(*[np.arange(0.0, size).astype(dtype) for size in volume_size], indexing='ij'), axis=-1).reshape(-1, dimension)

def transform_point(T, p):
    if len(T.shape) == 2:
        dim = len(p)
        homo_p = np.ones(dim + 1)
        homo_p[:dim] = p
        return (T@homo_p)[:dim]
    else:
        n, dim = p.shape
        homo_p = np.hstack((p, np.ones((n, 1))))
        transformed_homo_p = T @ homo_p[:, :, None]
        # print(transformed_homo_p)
        return transformed_homo_p[:, :dim, 0]
def transform_normal(T, n):
    if len(T.shape) == 2:
        dim = len(n)
        R = T[:dim, :dim]
        return R@n
    else:
        _, dim = n.shape
        R = T[:, :dim, :dim]
        return (R@n[:, :, None])[:, :, 0]
# 3d function
def project_point_to_line_seg(p, l1, l2):
    u = l2 - l1
    v = p - l1 
    if len(p.shape) == 1:
        beta = np.dot(u, v) / np.sum(u**2)
        alpha = 1 - beta
        projected_p = alpha * l1 + beta * l2 
        fall_on = (alpha >= 0) and (alpha <= 1) and (beta >= 0) and (beta <= 1)
        return fall_on, projected_p
    else:
        beta =  row_wise_dot(u, v) / np.sum(u**2, axis = 1)
        alpha = 1 - beta
        projected_p = alpha[:, None] * l1 + beta[:, None] * l2
        fall_on = np.logical_and(np.logical_and(alpha >= 0, alpha <= 1), np.logical_and(beta >= 0, beta <= 1))
        return fall_on, projected_p
# each side of the line
# 2d function
def point_to_line(p, l1, l2):
    if len(p.shape) == 1:
        d =  l1[0]* l2[1] + l1[1] * p[0] + l2[0] * p[1] - p[0]* l2[1] - l1[1]*l2[0] - l1[0] * p[1]
        return d
    else:
        d =  l1[:, 0]* l2[:, 1] + l1[:, 1] * p[:, 0] + l2[:, 0] * p[:, 1] - p[:, 0]* l2[:, 1] - l1[:, 1]*l2[:, 0] - l1[:, 0] * p[:, 1]
        return d
def point_in_triangle(p, t1, t2, t3):
    a_r = point_to_line(p, t1, t2) > 0
    b_r = point_to_line(p, t2, t3) > 0
    c_r = point_to_line(p, t3, t1) > 0
    if len(p.shape) == 1:
        return a_r == b_r and b_r == c_r
    else:
        return np.logical_and(a_r == b_r, b_r == c_r)
def project_point_to_triangle(p, t1, t2, t3):
    u = t2 - t1
    v = t3 - t1
    n = np.cross(u, v)
    w = p - t1
    if len(p.shape) == 1:
        gamma = np.dot(np.cross(u, w), n) / np.sum(n**2)
        beta = np.dot(np.cross(w, v), n) / np.sum(n**2)
        alpha = 1 - gamma - beta
        projected_p = alpha * t1 + beta * t2 + gamma * t3 
        fall_in = (alpha >= 0) and (alpha <= 1) and (beta >= 0) and (beta <= 1) and (gamma >= 0) and (gamma <= 1)
        return fall_in, projected_p
    else:
        gamma = row_wise_dot( np.cross(u, w), n ) / np.sum(n ** 2, axis = 1)
        beta = row_wise_dot( np.cross(w, v), n) / np.sum(n ** 2, axis = 1)
        alpha = 1 - gamma - beta
        projected_p = alpha[:, None] * t1 + beta[:, None] * t2 + gamma[:, None] * t3
        fall_in = np.logical_and(np.logical_and( np.logical_and(alpha >= 0, alpha <= 1), np.logical_and(beta >= 0, beta <= 1)),  np.logical_and(gamma >= 0, gamma <= 1) )
        return fall_in, projected_p

def get_sdf(kdtree, p, centers, normals, radius, z_axis = np.array([0.0, 0.0, 1.0])):
    _, ids = kdtree.query(p, k = 5)
    min_distance = 1e7
    line_seg_ids = set()
    num_p = len(centers)
    dim = len(p)
    for i in range(len(ids)):
        if ids[i] > 0:
            line_seg_ids.add(ids[i] - 1)
        if ids[i] < num_p - 1:
            line_seg_ids.add(ids[i])
    for lid in line_seg_ids:
        c1 = centers[lid]
        c2 = centers[lid + 1]
        n1 = normals[lid]
        n2 = normals[lid + 1]
        r1 = radius[lid]
        r2 = radius[lid + 1]

        _, projected_p = project_point_to_line_seg(p, c1, c2)
        # print(p - c1, c2 - c1, np.cross(p - c1, c2 - c1))
        n = normalized(np.cross(p - c1, c2 - c1))
        help_n = normalized(p - projected_p)
        # find two intersected points
        inter_p1 = None
        inter_p2 = None
        # point 1
        if dim == 3:
            local2global = np.eye(dim + 1)
            local2global[:dim, :dim] = rotation_matrix(z_axis, n1)
            local2global[:dim, dim] = c1
            global2local = LA.inv(local2global)
            tmp_p = transform_point(global2local, p)
            tmp_n = transform_normal(global2local, n)
            solution = np.zeros(dim)
            solution[0] = r1 / np.sqrt(1 + (tmp_n[0] * tmp_n[0]) / (tmp_n[1] * tmp_n[1]))
            solution[1] = -tmp_n[0] * solution[0] / tmp_n[1]
            tmp_help_n = transform_normal(global2local, help_n)
            if ( np.dot(solution, tmp_help_n) > 0) == (np.dot(tmp_p, tmp_help_n) > 0):
                inter_p1 = transform_point(local2global, solution)
            else:
                inter_p1 = transform_point(local2global, -solution)

            # point 2

            local2global = np.eye(dim + 1)
            local2global[:dim, :dim] = rotation_matrix(z_axis, n2)
            local2global[:dim, dim] = c2
            global2local = LA.inv(local2global)
            tmp_p = transform_point(global2local, p)
            tmp_n = transform_normal(global2local, n)
            solution = np.zeros(dim)
            solution[0] = r2 / np.sqrt(1 + (tmp_n[0] * tmp_n[0]) / (tmp_n[1] * tmp_n[1]))
            solution[1] = -tmp_n[0] * solution[0] / tmp_n[1]
            tmp_help_n = transform_normal(global2local, help_n)
            if ( np.dot(solution, tmp_help_n) > 0) == (np.dot(tmp_p, tmp_help_n) > 0):
                inter_p2 = transform_point(local2global, solution)
            else:
                inter_p2 = transform_point(local2global, -solution)

            sign = True
            fall_in_1, _ = project_point_to_triangle(p, inter_p1, c1, c2)
            fall_in_2, _ = project_point_to_triangle(p, c2, inter_p2, inter_p1)
            if fall_in_1 or fall_in_2:
                sign = False
            local_distance = 0
            fall_on, tmp_proj_p = project_point_to_line_seg(p, inter_p1, inter_p2)
            if fall_on:
                local_distance = LA.norm(p - tmp_proj_p)
            else:
                local_distance = np.min([LA.norm(p - inter_p1), LA.norm(p - inter_p2)])
            if not sign:
                local_distance = - local_distance
            if local_distance < min_distance:
                min_distance = local_distance
        # else:

    return min_distance


def get_sdf_batch(kdtree, query_points, centers, normals, radius, max_dist = 10, z_axis = np.array([0.0, 0.0, 1.0])):
    dist, ids = kdtree.query(query_points, k = 5)
    if max_dist < 0:
        query_selector = dist[:, 0] > max_dist
        sdf = np.ones(len(query_points)) * 1e7
    else:
        query_selector = dist[:, 0] < max_dist
        sdf = np.ones(len(query_points)) * max_dist
    
    batch_size, dim = query_points[query_selector].shape
    print("valid_voxel:", batch_size)
    
    weight = np.zeros(len(query_points))
    valid_query_points = query_points[query_selector]
    valid_sdf = sdf[query_selector]
    valid_ids = ids[query_selector]
    weight[query_selector] = 1.0
    valid_ids[valid_ids > 0] = valid_ids[valid_ids > 0] - 1
    help_z_axis = np.tile(z_axis, (batch_size, 1))
    

    for lid in range(valid_ids.shape[1]):
        l_start = valid_ids[:, lid]
        l_end = valid_ids[:, lid] + 1 

        c1 = centers[l_start]
        c2 = centers[l_end]
        n1 = normals[l_start]
        n2 = normals[l_end]
        r1 = radius[l_start]
        r2 = radius[l_end]
        if dim == 3:
            _, projected_p = project_point_to_line_seg(valid_query_points, c1, c2)
            n = normalized(np.cross(valid_query_points - c1, c2 - c1))
            help_n = normalized(valid_query_points - projected_p)
            # find two intersected points
            inter_p1 = None
            inter_p2 = None
            # point 1
            local2global = np.array([np.eye(dim + 1) for i in range(batch_size)])
            local2global[:, :dim, :dim] = rotation_matrix(help_z_axis, n1)
            local2global[:, :dim, dim] = c1
            global2local = LA.inv(local2global)
            tmp_p = transform_point(global2local, valid_query_points)
            tmp_n = transform_normal(global2local, n)
            solution = np.zeros((batch_size,  dim))
            solution[:, 0] = r1 / np.sqrt(1 + (tmp_n[:, 0] * tmp_n[:, 0]) / (tmp_n[:, 1] * tmp_n[:, 1]))
            solution[:, 1] = -tmp_n[:, 0] * solution[:, 0] / tmp_n[:, 1]
            tmp_help_n = transform_normal(global2local, help_n)
            selector = ( row_wise_dot( solution , tmp_help_n) > 0) != (row_wise_dot( tmp_p , tmp_help_n) > 0)
            solution[selector] = - solution[selector]
            inter_p1 = transform_point(local2global, solution)

            # point 2

            local2global = np.array([np.eye(dim + 1) for i in range(batch_size)])
            local2global[:, :dim, :dim] = rotation_matrix(help_z_axis, n2)
            local2global[:, :dim, dim] = c2
            global2local = LA.inv(local2global)
            tmp_p = transform_point(global2local, valid_query_points)
            tmp_n = transform_normal(global2local, n)
            solution = np.zeros((batch_size,  dim))
            solution[:, 0] = r2 / np.sqrt(1 + (tmp_n[:, 0] * tmp_n[:, 0]) / (tmp_n[:, 1] * tmp_n[:, 1]))
            solution[:, 1] = -tmp_n[:, 0] * solution[:, 0] / tmp_n[:, 1]
            tmp_help_n = transform_normal(global2local, help_n)
            selector = ( row_wise_dot( solution , tmp_help_n) > 0) != (row_wise_dot( tmp_p , tmp_help_n) > 0)
            solution[selector] = - solution[selector]
            inter_p2 = transform_point(local2global, solution)

            sign = np.ones(batch_size)
            fall_in_1, _ = project_point_to_triangle(valid_query_points, inter_p1, c1, c2)
            fall_in_2, _ = project_point_to_triangle(valid_query_points, c2, inter_p2, inter_p1)
            sign[np.logical_or (fall_in_1, fall_in_2)] = -1

            local_distance = np.zeros(batch_size)
            fall_on, tmp_proj_p = project_point_to_line_seg(valid_query_points, inter_p1, inter_p2)
            local_distance[fall_on] = LA.norm(valid_query_points[fall_on] - tmp_proj_p[fall_on], axis = 1)
            not_fall_on = np.logical_not(fall_on)
            local_distance[not_fall_on] = np.min(np.hstack( (LA.norm(valid_query_points[not_fall_on] - inter_p1[not_fall_on], axis = 1)[:,None], LA.norm(valid_query_points[not_fall_on] - inter_p2[not_fall_on], axis = 1)[:,None] )) , axis = 1)
            local_distance = local_distance * sign
            valid_sdf[valid_sdf > local_distance] = local_distance[valid_sdf > local_distance]
        else:
            direction1 = normalized(np.hstack( ((-n1[:, 1] / n1[:, 0])[:, None], np.ones((batch_size, 1)))))
            direction2 = normalized(np.hstack( ((-n2[:, 1] / n2[:, 0])[:, None], np.ones((batch_size, 1)))))
            selector = np.sum(direction1 * (valid_query_points - c1), axis = 1) < 0
            # print(selector)
            direction1[selector] = -direction1[selector]
            selector = np.sum(direction2 * (valid_query_points - c2), axis = 1) < 0
            direction2[selector] = -direction2[selector]
            inter_p1 = c1 + direction1 * np.tile(r1, (2, 1)).transpose()
            inter_p2 = c2 + direction2 * np.tile(r2, (2, 1)).transpose()

            sign = np.ones(batch_size)

            fall_in_1 = point_in_triangle(valid_query_points, inter_p1, c1, c2)
            fall_in_2 = point_in_triangle(valid_query_points, c2, inter_p2, inter_p1)
            sign[np.logical_or (fall_in_1, fall_in_2)] = -1

            local_distance = np.zeros(batch_size)
            fall_on, tmp_proj_p = project_point_to_line_seg(valid_query_points, inter_p1, inter_p2)
            local_distance[fall_on] = LA.norm(valid_query_points[fall_on] - tmp_proj_p[fall_on], axis = 1)
            not_fall_on = np.logical_not(fall_on)
            local_distance[not_fall_on] = np.min(np.hstack( (LA.norm(valid_query_points[not_fall_on] - inter_p1[not_fall_on], axis = 1)[:,None], LA.norm(valid_query_points[not_fall_on] - inter_p2[not_fall_on], axis = 1)[:,None] )) , axis = 1)
            local_distance = local_distance * sign
            valid_sdf[valid_sdf > local_distance] = local_distance[valid_sdf > local_distance]
    sdf[query_selector] = valid_sdf
    return sdf, weight

def contour2sdf(centers, radius, volume_shape, voxel_resolution = 1.0, max_dist = 10):
    num_p, dim = centers.shape
    normals = np.zeros_like(centers)
    # compute normals
    for i in range(num_p):    
        if i == 0: 
            normals[i] = normalized(centers[1] - centers[0])
        elif i == num_p - 1:
             normals[i] = normalized(centers[i] - centers[i-1])
        else:
            normals[i] = normalized(normalized(centers[i+1] - centers[i]) + normalized(centers[i] - centers[i-1]))
    kdtree = cKDTree(centers, 16)
    query_points = get_coordinates(volume_shape) + voxel_resolution / 2

    if max_dist < 0:
        return get_sdf_batch(kdtree, query_points, centers, normals, radius, max_dist = max_dist)
    # assume that there are not translation.
    # from voxel to pcd

    # reduce the size of query points
    kernel_size = int(max_dist / voxel_resolution)
    # from pcd to voxel

    center_voxel = (centers / voxel_resolution).astype(int)
    # print(np.stack(np.meshgrid(*[np.arange(0, size) for size in (kernel_size * np.ones(dim))], indexing='ij'), axis=-1).reshape(-1, dim))
    kernel =  np.stack(np.meshgrid(*[np.arange(0, size) for size in (kernel_size * np.ones(dim))], indexing='ij'), axis=-1).reshape(-1, dim) - kernel_size / 2

    counted_voxel = (np.repeat(center_voxel, len(kernel), axis = 0).reshape(num_p, len(kernel), dim) + kernel).reshape(-1, dim)
    counted_voxel[counted_voxel < 0] = 0
    for i in range(dim):
        counted_voxel[counted_voxel[:, i] >= volume_shape[i], i] = volume_shape[i] - 1

    if dim == 2:
        counted_vid = counted_voxel[:, 0] * volume_shape[1] + counted_voxel[:, 1]
    elif dim == 3:
        counted_vid = counted_voxel[:, 0] * (volume_shape[1] * volume_shape[2]) + counted_voxel[:, 1] * volume_shape[2] + counted_voxel[:, 2]
    # print(len(counted_vid))
    counted_vid = np.array(list(set(counted_vid.astype(int))))
    # print(len(counted_vid))
    sdf = np.ones(len(query_points)) * max_dist
    weight = np.zeros(len(query_points))
    sdf[counted_vid], weight[counted_vid] = get_sdf_batch(kdtree, query_points[counted_vid], centers, normals, radius, max_dist = max_dist)
    
    return sdf.reshape(volume_shape), weight.reshape(volume_shape)

def generate_raw_data(label, positive = [128, 255], negative = [32, 160], noise_mean = [-1, 1], noise_std = [-15, 30]):
    raw_data = np.zeros_like(label)
    n_mean = np.random.rand() * 2 - 1
    n_std = np.random.rand() * 24
    raw_data[label == 1] = np.random.randint(low = positive[0], high = positive[1], size = raw_data[label == 1].shape)
    raw_data[label == 0] = np.random.randint(low = negative[0], high = negative[1], size = raw_data[label == 0].shape)
    raw_data = raw_data + np.random.normal(n_mean, n_std, raw_data.shape)
    return raw_data.astype(int)
def save_itk(filename, imageArray, origin = None, spacing = None):
    itkimage = sitk.GetImageFromArray(imageArray)
    if origin is not None:
        itkimage.SetOrigin(origin)
    if spacing is not None:
        itkimage.SetSpacing(spacing)
    sitk.WriteImage(itkimage, filename, useCompression = True)
if __name__ == "__main__":
    import time    
    for id in range(8):
        filename_data ='/home/wlsdzyzl/project/topohpm/generated_files/synthetic_dataset/val/data/' + str(id) + '.nii.gz'
        filename_label ='/home/wlsdzyzl/project/topohpm/generated_files/synthetic_dataset/val/label/' + str(id) + '.nii.gz'
        time_start=time.time()
        volume_size = (128, 128, 128)
        start_point = np.zeros(3)
        start_point[0] = np.random.normal(48.0, 5.0)
        start_point[1] = np.random.normal(48.0, 5.0)
        start_point[2] = np.random.normal(100, 5.0)



        num_mass_p = 8
        len_mean = volume_size[-1] * 0.9 / num_mass_p
        num_all_p = 500
        start_r = 5
        end_r = 1
        max_dist = 10

        mass_p = generate_mass_points(np.array(start_point), angle_range = np.pi / 4, num = num_mass_p, len_mean = len_mean)
        mass_r, radius = generate_radius(num_mass_p + 1, num_all_p, start_r, 1, end_r, 0.25)
        radius[radius <= 0.2] = 0.2 
        # print(radius)
        # root
        curve_p = cubic_spline(mass_p, num_all_p)
        sdf_volume, weight_volume = contour2sdf(curve_p, radius, volume_size, max_dist = max_dist)
        # leaf

        
        rand_r = 1
        mass_p_1 = generate_mass_points(mass_p[2].copy(), angle_range = np.pi / 4, num = int(num_mass_p * 0.5), len_mean = len_mean , init_direction = normalized(mass_p[3] - mass_p[1]))
        mass_r_1, radius_1 = generate_radius(int(num_mass_p * 0.5) + 1, num_all_p, mass_r[3] * rand_r, 0.8, end_r, 0.25)
        curve_p_1 = cubic_spline(mass_p_1, num_all_p)
        
        sdf_volume_1, weight_volume_1 = contour2sdf(curve_p_1, radius_1, volume_size, max_dist = max_dist)

        time_end=time.time()
        print('time cost',(time_end-time_start) / 60,'min')

        sdf_volume[sdf_volume > sdf_volume_1] = sdf_volume_1[sdf_volume > sdf_volume_1]


        # second curve:
        start_point[0] = np.random.normal(80.0, 5.0)
        start_point[1] = np.random.normal(80.0, 5.0)
        start_point[2] = np.random.normal(100, 5.0)


        mass_p = generate_mass_points(np.array(start_point), angle_range = np.pi / 4, num = num_mass_p, len_mean = len_mean)
        mass_r, radius = generate_radius(num_mass_p + 1, num_all_p, start_r, 1, end_r, 0.25)
        radius[radius <= 0.2] = 0.2
        # root
        curve_p = cubic_spline(mass_p, num_all_p)
        sdf_volume_2, weight_volume_2 = contour2sdf(curve_p, radius, volume_size, max_dist = max_dist)

        sdf_volume[sdf_volume > sdf_volume_2] = sdf_volume_2[sdf_volume > sdf_volume_2]

        label = (sdf_volume < 0).astype('uint8')

        raw = generate_raw_data(label)
        save_itk(filename_data, raw)
        save_itk(filename_label, label)

