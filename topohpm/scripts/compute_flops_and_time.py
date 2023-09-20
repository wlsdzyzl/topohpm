from topohpm.scripts.utils import *
from thop import profile

import random

import torch

from topohpm.unet3d.config import _load_config_yaml as load_config
from topohpm.regularized_unet3d.trainer import create_trainer
import time

def main(configfile):
    # Load and log experiment configuration
    config = load_config(configfile)
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
    # create trainer
    trainer = create_trainer(config)
    # Start training
    start = time.time()
    trainer.fit()
    end = time.time()
    return end - start
## 2d test
## warm-up
# main('test_config/test_unet.yml')
# time_dict = {'unet':main('test_config/test_unet.yml'),
#     'cldice': main('test_config/test_cldice.yml'),
#     'naive': main('test_config/test_naive.yml'),
#     'cubical': main('test_config/test_cubical.yml'),
#     'skeleton': main('test_config/test_skeleton.yml'),}
# print(time_dict)

## 3d test
# warm-up
main('test_config/test_unet_3d.yml')
time_dict_3d = {'unet3d':main('test_config/test_unet_3d.yml'),
    'cldice3d': main('test_config/test_cldice_3d.yml'),
    'naive3d': main('test_config/test_naive_3d.yml'),
    'cubical3d': main('test_config/test_cubical_3d.yml'),
    'skeleton3d': main('test_config/test_skeleton_3d.yml'),}
print(time_dict_3d)