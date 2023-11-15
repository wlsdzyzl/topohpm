#!/bin/bash

train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-unet/train_config_none_1.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-unet/train_config_none_2.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-unet/train_config_none_3.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-unet/train_config_none_4.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-unet/train_config_none_5.yml