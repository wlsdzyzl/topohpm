
import numpy as np
from contour_generation import *
from contour2sdf import *
import imageio

volume_size = (500, 500)
start_point = np.zeros(2)
start_point[0] = np.random.normal(250.0, 10.0)
start_point[1] = np.random.normal(450.0, 10.0)



num_mass_p = 8
len_mean = volume_size[-1] * 0.85 / num_mass_p
num_all_p = 500
start_r = 5
end_r = 1
max_dist = 10

branch_id = 3

mass_p = generate_mass_points(np.array(start_point), angle_range = np.pi / 4, num = num_mass_p, len_mean = len_mean)
mass_r, radius = generate_radius(len(mass_p), num_all_p, start_r, 1, end_r, 0.25)
radius[radius <= 0.2] = 0.2 
# print(radius)
# root
curve_p = cubic_spline(mass_p, num_all_p)
sdf_volume, weight_volume = contour2sdf(curve_p, radius, volume_size, max_dist = max_dist)
# leaf


rand_r = 1
mass_p_1 = generate_mass_points(mass_p[branch_id].copy(), angle_range = np.pi / 4, num = int(num_mass_p * 0.5), len_mean = len_mean , init_direction = normalized(mass_p[branch_id + 1] - mass_p[branch_id - 1]))
mass_r_1, radius_1 = generate_radius(len(mass_p_1), num_all_p, mass_r[branch_id] * rand_r, 0.1, end_r, 0.25)
curve_p_1 = cubic_spline(mass_p_1, num_all_p)

sdf_volume_1, weight_volume_1 = contour2sdf(curve_p_1, radius_1, volume_size, max_dist = max_dist)


sdf_volume[sdf_volume > sdf_volume_1] = sdf_volume_1[sdf_volume > sdf_volume_1]


# # second curve:
# start_point[0] = np.random.normal(80.0, 5.0)
# start_point[1] = np.random.normal(80.0, 5.0)
# start_point[2] = np.random.normal(100, 5.0)


# mass_p = generate_mass_points(np.array(start_point), angle_range = np.pi / 4, num = num_mass_p, len_mean = len_mean)
# mass_r, radius = generate_radius(num_mass_p + 1, num_all_p, start_r, 1, end_r, 0.5)
# radius[radius <= 0.2] = 0.2
# # root
# curve_p = cubic_spline(mass_p, num_all_p)
# sdf_volume_2, weight_volume_2 = contour2sdf(curve_p, radius, volume_size, max_dist = max_dist)

# sdf_volume[sdf_volume > sdf_volume_2] = sdf_volume_2[sdf_volume > sdf_volume_2]

label = (sdf_volume < 0).astype('uint8')
raw = generate_raw_data(label)

imageio.imwrite('data.png', raw.astype('uint8'))
imageio.imwrite('label.png', label * 255)


# visualization
if False:
    c1 = ['tab:blue' for i in range(len(mass_p))]
    c2 = ['tab:blue' for i in range(len(mass_p_1))]
    c1[branch_id] = 'tab:orange'
    c2[0] = 'tab:orange' 
    plt.scatter(mass_p[:,0], mass_p[:,1], c=c1, s = mass_r * 22)
    plt.scatter(mass_p_1[:,0], mass_p_1[:,1], c=c2, s = mass_r_1 * 22)


    plt.xlim(0, volume_size[0])
    plt.ylim(0, volume_size[1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.matshow(sdf_volume)
    plt.axis('off')
    plt.savefig('point.png')
    plt.savefig('point.svg', format = 'svg', dpi=300)

    plt.clf()

    plt.scatter(mass_p[:,0], mass_p[:,1], c=c1, s = mass_r * 22)
    plt.scatter(mass_p_1[:,0], mass_p_1[:,1], c=c2, s = mass_r_1 * 22)

    plt.scatter(curve_p[:,0], curve_p[:,1], c = 'brown', s = 1)
    plt.scatter(curve_p_1[:,0], curve_p_1[:,1], c = 'brown', s = 1)

    plt.xlim(0, volume_size[0])
    plt.ylim(0, volume_size[1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.matshow(sdf_volume)
    plt.axis('off')
    plt.savefig('curve.png')
    plt.savefig('curve.svg', format = 'svg', dpi=300)

    plt.clf()


    # print(sdf_volume[sdf_volume < 0])
    # visualization
    query_points = get_coordinates(volume_size) + 0.5
    plt.scatter(query_points[:,0], query_points[:,1], c = sdf_volume.reshape(-1), s = 1)
    plt.xlim(0, volume_size[0])
    plt.ylim(0, volume_size[1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.matshow(sdf_volume)

    plt.axis('off')
    plt.savefig('sdf.png')

    plt.clf()

    sdf_volume[sdf_volume > 0] = 0
    sdf_volume[sdf_volume < 0] = 1
    # print(sdf_volume[sdf_volume < 0])
    # visualization
    query_points = get_coordinates(volume_size) + 0.5
    plt.scatter(query_points[:,0], query_points[:,1], c = sdf_volume.reshape(-1), s = 1)
    plt.xlim(0, volume_size[0])
    plt.ylim(0, volume_size[1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.matshow(sdf_volume)
    plt.axis('off')
    plt.savefig('mask.png')


    plt.clf()

    query_points = get_coordinates(volume_size) + 0.5
    plt.scatter(query_points[:,0], query_points[:,1], c = raw.reshape(-1), s = 1)
    plt.xlim(0, volume_size[0])
    plt.ylim(0, volume_size[1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.matshow(sdf_volume)
    plt.axis('off')
    plt.savefig('raw.png')