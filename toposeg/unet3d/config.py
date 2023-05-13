import argparse

import torch
import yaml

from toposeg.unet3d.utils import get_logger

logger = get_logger('ConfigLoader')

def _load_config_yaml(config_file):
    config = yaml.safe_load(open(config_file, 'r'))
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warning('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config
    
def load_config():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    return _load_config_yaml(args.config)


