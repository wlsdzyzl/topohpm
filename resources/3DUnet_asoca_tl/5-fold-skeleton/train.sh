#!/bin/bash


train3dunet --config /home/wlsdzyzl/project/topohpm/resources/3DUnet_imagecas/5-fold-skeleton/train_config_skeletonloss_1.yml
train3dunet --config /home/wlsdzyzl/project/topohpm/resources/3DUnet_imagecas/5-fold-skeleton/train_config_skeletonloss_2.yml
train3dunet --config /home/wlsdzyzl/project/topohpm/resources/3DUnet_imagecas/5-fold-skeleton/train_config_skeletonloss_3.yml
train3dunet --config /home/wlsdzyzl/project/topohpm/resources/3DUnet_imagecas/5-fold-skeleton/train_config_skeletonloss_4.yml
train3dunet --config /home/wlsdzyzl/project/topohpm/resources/3DUnet_imagecas/5-fold-skeleton/train_config_skeletonloss_5.yml