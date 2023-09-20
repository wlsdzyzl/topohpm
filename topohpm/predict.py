import importlib
import os

import torch
import torch.nn as nn

from topohpm.datasets.utils import get_test_loaders
from topohpm.unet3d.utils import get_logger, load_checkpoint
from topohpm.unet3d.config import load_config
from topohpm.unet3d.model import get_model

logger = get_logger('UNet3DPredict')


def _get_predictor(model, output_dir, config, output_seg_dir = None, threshold = 0.5):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardH5Predictor')

    m = importlib.import_module('topohpm.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, output_seg_dir, threshold, **predictor_config)


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    load_checkpoint(model_path, model)
    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    output_seg_dir = config['loaders'].get('output_seg_dir', None)
    threshold = config['loaders'].get('threshold', 0.5)
    if output_seg_dir is not None:
        os.makedirs(output_seg_dir, exist_ok=True)
        logger.info(f'Saving segments to: {output_seg_dir}')
    # create predictor instance
    predictor = _get_predictor(model, output_dir, config, output_seg_dir, threshold)

    for test_loader in get_test_loaders(config):
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader)


if __name__ == '__main__':
    main()