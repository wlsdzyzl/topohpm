import torch, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn
import imageio

import sys, getopt
import os
import glob


from skimage.morphology import skeletonize
from scipy import ndimage, misc
import gudhi as gd

pred_path = "/home/wlsdzyzl/project/topohpm/topohpm/scripts/skeleton_loss_demo/pred.png"
label_path = "/home/wlsdzyzl/project/topohpm/topohpm/scripts/skeleton_loss_demo/gt.png"

output_dir = '/home/wlsdzyzl/project/topohpm/topohpm/scripts/skeleton_loss_demo'

pred_img = ndimage.zoom( np.asarray(imageio.imread(pred_path)) , (0.5, 0.5, 1))
label_img = ndimage.zoom( np.asarray(imageio.imread(label_path)) , (0.5, 0.5, 1))

pred_img[pred_img <= 10] = 10
pred_img[pred_img >= 245] = 245

noise =  np.random.randint(low = -10, high = 10, size = (pred_img.shape[0], pred_img.shape[1]))
pred_img[:, :, 0] = pred_img[:, :, 0] + noise
pred_img[:, :, 1] = pred_img[:, :, 1] + noise
pred_img[:, :, 2] = pred_img[:, :, 2] + noise


pred_img[np.logical_and(pred_img >= 0.6 * 255, label_img >= 0.6*255)] = 255
pred_img[np.logical_and(pred_img < 0.6 * 255, label_img < 0.6*255)] = 0
### skeleton loss
# to binary
pred = pred_img[:, :, 0] < 128
label = label_img[:, :, 0] < 128

label_sk = skeletonize(label)
pred_sk = skeletonize(pred)

pos_tmp = np.logical_and(label_sk, pred)
neg_tmp = np.logical_and(pred_sk, label)

positive = np.logical_xor(pos_tmp, label_sk)
negative = np.logical_xor(neg_tmp, pred_sk)

fig, ax = plt.subplots()
ax.matshow(1 - pred, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, pred.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, pred.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/pred_grids.png', dpi=dpi)

plt.clf()

fig, ax = plt.subplots()
ax.matshow(1 - label, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/gt_grids.png', dpi=dpi)

plt.clf()

fig, ax = plt.subplots()
ax.matshow(1 - pred_sk, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/pred_sk_grids.png', dpi=dpi)

plt.clf()

fig, ax = plt.subplots()
ax.matshow(1 - label_sk, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/gt_sk_grids.png', dpi=dpi)


plt.clf()

fig, ax = plt.subplots()
ax.matshow(1 - positive, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/positive_grids.png', dpi=dpi)

plt.clf()

fig, ax = plt.subplots()
ax.matshow(1 - negative, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/negative_grids.png', dpi=dpi)


plt.clf()

fig, ax = plt.subplots()
ax.matshow(1 - pos_tmp, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/postmp_grids.png', dpi=dpi)

plt.clf()

fig, ax = plt.subplots()
ax.matshow(1 - neg_tmp, cmap='gray')
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/negtmp_grids.png', dpi=dpi)


plt.clf()


pred_img_sk = pred_img.copy()


pred_img_sk[:, :, 0][positive]= 0
pred_img_sk[:, :, 1][positive]= 255
pred_img_sk[:, :, 2][positive]= 0

pred_img_sk[:, :, 0][negative]= 255
pred_img_sk[:, :, 1][negative]= 0
pred_img_sk[:, :, 2][negative]= 0
print(pred_img_sk[positive][:,0])

fig, ax = plt.subplots()
ax.imshow(pred_img_sk)
ax.set_aspect(1)
# ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
# ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
# ax.yaxis.grid(True)
# ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/final_grids_sk.png', dpi=dpi)

#### naive loss

positive = np.logical_and(label, np.logical_not(pred))
negative = np.logical_and(np.logical_not(label), pred)


plt.clf()


pred_img_nv = pred_img.copy()


pred_img_nv[:, :, 0][positive]= 0
pred_img_nv[:, :, 1][positive]= 255
pred_img_nv[:, :, 2][positive]= 0

pred_img_nv[:, :, 0][negative]= 255
pred_img_nv[:, :, 1][negative]= 0
pred_img_nv[:, :, 2][negative]= 0
print(pred_img_nv[positive][:,0])

fig, ax = plt.subplots()
ax.imshow(pred_img_nv)
ax.set_aspect(1)
# ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
# ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
# ax.yaxis.grid(True)
# ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
fig.savefig(fname = output_dir + '/final_grids_nv.png', dpi=dpi)


#### cubical loss

def detect_critical_points_gudhi(likelihood, label, dims = [0]):
    """
    Compute the critical points of the image (Value range from 0 -> 1)
    Args:
        likelihood: Likelihood image from the output of the neural networks
    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.
    """
    # likelihood[ np.logical_and(likelihood > 0.6, label > 0.5 )] = 1.0
    # likelihood[ np.logical_and(likelihood < 0.6, label < 0.5 )] = 0.0

    lh = 1.0 - likelihood
    lh_vector = np.asarray(lh).flatten()
    lh_cubic = gd.CubicalComplex(
        dimensions=[lh.shape[0], lh.shape[1]],
        top_dimensional_cells=lh_vector
    )

    Diag_lh = lh_cubic.persistence(homology_coeff_field=2, min_persistence=0)
    # print(Diag_lh)
    pairs_lh = lh_cubic.cofaces_of_persistence_pairs()

    # If the paris is 0, return False to skip
    if (len(pairs_lh[0])==0): return [], []
    # print(pairs_lh)
    # return persistence diagram, birth/death critical points
    positive_idx = np.full(likelihood.shape, False)
    negative_idx = np.full(likelihood.shape, False)
    birth_ids = []
    death_ids = []
    for dim in dims:
        bd_pairs = pairs_lh[0][dim]
        birth_id = np.hstack( ((bd_pairs[:, 0] // likelihood.shape[1])[:, None], (bd_pairs[:, 0] % likelihood.shape[1])[:, None] ))
        death_id = np.hstack( ((bd_pairs[:, 1] // likelihood.shape[1])[:, None], (bd_pairs[:, 1] % likelihood.shape[1])[:, None] ))
        overall_id = np.vstack((birth_id, death_id)).astype(int)
        birth_ids.append(birth_id)
        death_ids.append(death_id)
        # print(label.shape)
        # print(overall_id)
        positive_selector = np.logical_and(label[overall_id[:, 0], overall_id[:, 1]] > 0.5, likelihood[overall_id[:, 0], overall_id[:, 1]] < 0.5)
        negative_selector = np.logical_and(label[overall_id[:, 0], overall_id[:, 1]] < 0.5, likelihood[overall_id[:, 0], overall_id[:, 1]] > 0.5)
        tmp_positive_idx = overall_id[positive_selector]
        tmp_negative_idx = overall_id[negative_selector]
        positive_idx[tmp_positive_idx[:, 0], tmp_positive_idx[:, 1]] = True
        negative_idx[tmp_negative_idx[:, 0], tmp_negative_idx[:, 1]] = True
    return positive_idx, negative_idx, birth_ids, death_ids

pred = (1 - pred_img[:, :, 0] / 255.0).transpose()
label = (label_img[:, :, 0] < 128).astype(float).transpose()

positive_t, negative_t, birth_ids, death_ids = detect_critical_points_gudhi(pred, label)
positive = positive_t.transpose()
negative = negative_t.transpose()
plt.clf()

# tmp_id = 0
# for bid in birth_ids[0]:
#     tmp_pred = pred.copy()
#     threshold = tmp_pred[bid[0], bid[1]]
#     print(threshold)
#     tmp_pred[tmp_pred >= threshold] = 1.0
#     tmp_pred[tmp_pred < threshold] = 0.0
#     imageio.imwrite(output_dir + '/{}.png'.format(tmp_id), (1 - tmp_pred).transpose())
#     tmp_id += 1

# for did in death_ids[0]:
#     tmp_pred = pred.copy()
#     threshold = tmp_pred[did[0], did[1]]
#     print(threshold)
#     tmp_pred[tmp_pred >= threshold] = 1.0
#     tmp_pred[tmp_pred < threshold] = 0.0
#     imageio.imwrite(output_dir + '/{}.png'.format(tmp_id), (1 - tmp_pred).transpose())
#     tmp_id += 1

pred_img_cl = pred_img.copy()


# pred_img_cl[:, :, 0][positive]= 0
# pred_img_cl[:, :, 1][positive]= 255
# pred_img_cl[:, :, 2][positive]= 0

# pred_img_cl[:, :, 0][negative]= 255
# pred_img_cl[:, :, 1][negative]= 0
# pred_img_cl[:, :, 2][negative]= 0
# print(pred_img_cl[positive][:,0])

fig, ax = plt.subplots()

# plt.rc('legend', fontsize=15)    # legend fontsize
ax.imshow(pred_img_cl)
ax.set_aspect(1)
ax.set_yticks(np.arange(0, label.shape[1]) - 0.5, minor=False)
ax.set_xticks(np.arange(0, label.shape[0]) - 0.5, minor=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
dpi=200
ax.scatter(birth_ids[0][:, 0], birth_ids[0][:, 1], label='birth points', zorder=10, s=20)
ax.scatter(death_ids[0][:, 0], death_ids[0][:, 1], marker="^", label = 'death points', zorder=10, s=20)
ax.legend(framealpha=1.0).set_zorder(20)
fig.savefig(fname = output_dir + '/final_grids_cl.png', dpi=dpi)


## generate raw data
def generate_raw_data(pred, label, positive = [144, 255], negative = [32, 150], noise_mean = [-0.5, 0.5], noise_std = [-15, 30]):
    raw_data = np.zeros_like(label)
    raw_data[label == 1] = np.random.randint(low = positive[0], high = positive[1], size = raw_data[label == 1].shape)
    raw_data[label == 0] = np.random.randint(low = negative[0], high = negative[1], size = raw_data[label == 0].shape)
    raw_data = raw_data + (pred - label) * np.random.normal(80, 50, raw_data.shape)
    raw_data[raw_data > 255] = 255
    raw_data[raw_data < 0 ] = 0
    return raw_data.astype(int)

raw_img = ndimage.zoom( generate_raw_data(pred.T, label.T) , (5.0, 5.0))
imageio.imwrite(output_dir + '/raw_image.png', raw_img)