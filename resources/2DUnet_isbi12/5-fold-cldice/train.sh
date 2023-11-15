#!/bin/bash
train3dunet --config /home/*****/project/toposeg/resources/2DUnet_isbi12/5-fold-cldice/train_config_cldiceloss_1.yml
train3dunet --config /home/*****/project/toposeg/resources/2DUnet_isbi12/5-fold-cldice/train_config_cldiceloss_2.yml
train3dunet --config /home/*****/project/toposeg/resources/2DUnet_isbi12/5-fold-cldice/train_config_cldiceloss_3.yml
train3dunet --config /home/*****/project/toposeg/resources/2DUnet_isbi12/5-fold-cldice/train_config_cldiceloss_4.yml
train3dunet --config /home/*****/project/toposeg/resources/2DUnet_isbi12/5-fold-cldice/train_config_cldiceloss_5.yml
