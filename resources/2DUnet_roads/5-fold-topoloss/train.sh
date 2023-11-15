#!/bin/bash
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/2DUnet_roads/5-fold-topoloss/train_config_topoloss_1.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/2DUnet_roads/5-fold-topoloss/train_config_topoloss_2.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/2DUnet_roads/5-fold-topoloss/train_config_topoloss_3.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/2DUnet_roads/5-fold-topoloss/train_config_topoloss_4.yml
train3dunet --config /home/wlsdzyzl/project/toposeg/resources/2DUnet_roads/5-fold-topoloss/train_config_topoloss_5.yml
