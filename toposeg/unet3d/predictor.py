import os

import h5py
import SimpleITK as sitk
import numpy as np
import torch
from skimage import measure

from toposeg.datasets.nii import AbstractNIIDataset
from toposeg.datasets.img import IMGDataset
from toposeg.datasets.utils import SliceBuilder
from toposeg.unet3d.utils import get_logger
from toposeg.unet3d.utils import remove_halo
import imageio
logger = get_logger('UNetPredictor')

# changed by *****
def _get_output_file(dataset, suffix='_prediction', output_dir=None):
    input_dir, file_name = os.path.split(dataset.file_path)
    if output_dir is None:
        output_dir = input_dir
    path, ext = os.path.splitext(file_name)
    path, add_ext = os.path.splitext(path)
    if add_ext != '':
        ext = add_ext + '.' + ext
    output_file = os.path.join(output_dir, path + suffix)
    return output_file


class _AbstractPredictor:
    def __init__(self, model, output_dir, config, output_seg_dir = None, threshold = 0.5, **kwargs):
        self.model = model
        self.output_dir = output_dir
        self.config = config
        self.predictor_config = kwargs
        self.output_seg_dir = output_seg_dir
        self.threshold = threshold
    @staticmethod
    def volume_shape(dataset):
        raw = dataset.raw
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def __call__(self, test_loader):
        raise NotImplementedError


# Added By *****
class StandardNIIPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result as H5 file.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `dest_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        output_dir (str): path to the output directory (optional)
        config (dict): global config dict
    """

    def __init__(self, model, output_dir, config, output_seg_dir = None, threshold = 0.5, **kwargs):
        super().__init__(model, output_dir, config, output_seg_dir, threshold, **kwargs)

    def __call__(self, test_loader):
        
        assert isinstance(test_loader.dataset, AbstractNIIDataset), 'dataset need to be in H5 format.'

        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
        output_file = _get_output_file(dataset=test_loader.dataset, suffix = '',output_dir=self.output_dir)
        if self.output_seg_dir is not None:
            output_seg_file = _get_output_file(dataset=test_loader.dataset, suffix = '', output_dir=self.output_seg_dir)
        else:
            output_seg_file = None
        out_channels = self.config['model'].get('out_channels')

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(test_loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = self.volume_shape(test_loader.dataset)
        # note that the shape of output
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        patch_halo = self.predictor_config.get('patch_halo', (4, 8, 8))
        self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
        logger.info(f'Using patch_halo: {patch_halo}')


        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads)

        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in test_loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        logger.info(f'Saving predictions for slice:{index}...')

                        # remove halo in order to avoid block artifacts in the output probability maps
                        u_prediction, u_index = remove_halo(pred, index, volume_shape, patch_halo)
                        # accumulate probabilities into the output prediction array
                        prediction_map[u_index] += u_prediction
                        # count voxel visits for normalization
                        normalization_mask[u_index] += 1

        # save results
        logger.info(f'Saving predictions to: {output_file}.nii.gz')
        self._save_results(prediction_maps, normalization_masks, output_heads, output_file, output_seg_file, test_loader.dataset)

    def _allocate_prediction_maps(self, output_shape, output_heads):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, output_seg_file, dataset):
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)
        idxs = [ str(i) for i in range(output_heads)]
        # save probability maps
        for prediction_map, normalization_mask, idx in zip(prediction_maps, normalization_masks,
                                                                          idxs):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding is not None:
                z_s, y_s, x_s = [_slice_from_pad(p) for p in dataset.mirror_padding]

                logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')

                prediction_map = prediction_map[:, z_s, y_s, x_s]
            if self.predictor_config.get('squeeze_if_one_channel', True) and prediction_map.ndim > 3 and prediction_map.shape[0] == 1:
                prediction_map = np.squeeze(prediction_map)
            output_image = sitk.GetImageFromArray(prediction_map)
            # print(prediction_map.shape)
            if len(idxs) > 1:
                sitk.WriteImage(output_image, output_file + idx + '.nii.gz', True)
            else:
                sitk.WriteImage(output_image, output_file + '.nii.gz', True)
            if output_seg_file is not None:
                output_seg = sitk.GetImageFromArray((prediction_map > self.threshold).astype('uint8'))
                if len(idxs) > 1:
                    sitk.WriteImage(output_seg, output_seg_file + idx + '.nii.gz', True)
                else:
                    sitk.WriteImage(output_seg, output_seg_file + '.nii.gz', True)
    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"


class IMGPredictor(_AbstractPredictor):
    """
    Applies the model on the given dataset and saves the result as H5 file.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.

    The output dataset names inside the H5 is given by `dest_dataset_name` config argument. If the argument is
    not present in the config 'predictions{n}' is used as a default dataset name, where `n` denotes the number
    of the output head from the network.

    Args:
        model (Unet2D): trained 3D UNet model used for prediction
        output_dir (str): path to the output directory (optional)
        config (dict): global config dict
    """

    def __init__(self, model, output_dir, config, output_seg_dir = None, threshold = 0.5, **kwargs):
        super().__init__(model, output_dir, config, output_seg_dir, threshold, **kwargs)

    def __call__(self, test_loader):
        
        assert isinstance(test_loader.dataset, IMGDataset), 'dataset need to be in img format.'

        logger.info(f"Processing '{test_loader.dataset.file_path}'...")

        output_file = _get_output_file(dataset=test_loader.dataset, suffix = '',output_dir=self.output_dir)
        if self.output_seg_dir is not None:
            output_seg_file = _get_output_file(dataset=test_loader.dataset, suffix = '', output_dir=self.output_seg_dir)
        else:
            output_seg_file = None

        out_channels = self.config['model'].get('out_channels')
        # output channel should be one
        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        logger.info(f'Running prediction on {len(test_loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = self.volume_shape(test_loader.dataset)
        if volume_shape[0] != 1:
            print('2D Unet, the first dim should be 1.')
        # note that the shape of output
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        patch_halo = self.predictor_config.get('patch_halo', (0, 8, 8))
        self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
        logger.info(f'Using patch_halo: {patch_halo}')


        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads)

        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in test_loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        logger.info(f'Saving predictions for slice:{index}...')

                        # remove halo in order to avoid block artifacts in the output probability maps
                        u_prediction, u_index = remove_halo(pred, index, volume_shape, patch_halo)
                        # accumulate probabilities into the output prediction array
                        prediction_map[u_index] += u_prediction
                        # count voxel visits for normalization
                        normalization_mask[u_index] += 1

        # save results
        logger.info(f'Saving predictions to: {output_file}.nii.gz')
        self._save_results(prediction_maps, normalization_masks, output_heads, output_file, output_seg_file, test_loader.dataset)

    def _allocate_prediction_maps(self, output_shape, output_heads):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, output_seg_file, dataset):
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)
        idxs = [ str(i) for i in range(output_heads)]
        # save probability maps
        for prediction_map, normalization_mask, idx in zip(prediction_maps, normalization_masks,
                                                                          idxs):
            prediction_map = prediction_map / normalization_mask

            if dataset.mirror_padding is not None:
                z_s, y_s, x_s = [_slice_from_pad(p) for p in dataset.mirror_padding]

                logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')

                prediction_map = prediction_map[:, z_s, y_s, x_s]
            if self.predictor_config.get('squeeze_if_one_channel', True) and prediction_map.ndim > 3 and prediction_map.shape[0] == 1:
                prediction_map = np.squeeze(prediction_map)
            # print(prediction_map.shape)
            if len(idxs) > 1:
                    imageio.imwrite(output_file + idx + '.png', (prediction_map * 255).astype('uint8'))
            else:
                    imageio.imwrite(output_file + '.png', (prediction_map * 255).astype('uint8'))

            if output_seg_file is not None:
                if len(idxs) > 1:
                    imageio.imwrite(output_seg_file + idx +'.png', (prediction_map > self.threshold).astype('uint8') * 255)
                else:
                    imageio.imwrite(output_seg_file + '.png', (prediction_map > self.threshold).astype('uint8') * 255)


    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"

