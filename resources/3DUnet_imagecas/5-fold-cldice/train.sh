#!/bin/bash
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-cldice/train_config_cldiceloss_1.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-cldice/train_config_cldiceloss_2.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-cldice/train_config_cldiceloss_3.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-cldice/train_config_cldiceloss_4.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/3DUnet_imagecas/5-fold-cldice/train_config_cldiceloss_5.yml
