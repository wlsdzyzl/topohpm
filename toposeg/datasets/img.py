import collections
import os
import glob
from itertools import chain

import imageio
import numpy as np
import torch

from toposeg.augment import transforms
from toposeg.datasets.utils import ConfigDataset, calculate_stats, get_slice_builder
from toposeg.unet3d.utils import get_logger

logger = get_logger('IMGDataset')

class IMGDataset(ConfigDataset):
    def __init__(self, raw_file_path, label_file_path, mask_file_path, phase, slice_builder_config, transformer_config, mirror_padding=(0, 32, 32), expand_dims=True):
        assert phase in ['train', 'val', 'test']

        # use mirror padding only during the 'test' phase
        if phase in ['train', 'val']:
            mirror_padding = None
        if mirror_padding is not None:
            assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        self.mirror_padding = mirror_padding

        self.phase = phase

        # load raw images
        self.raw = self._load_file(raw_file_path, expand_dims)
        self.file_path = raw_file_path
        stats = calculate_stats(self.raw)

        transformer = transforms.Transformer(transformer_config, stats)
        # load raw images transformer
        self.raw_transform = transformer.raw_transform()
        self.mask = None
        if phase != 'test':
            # load labeled images
            self.label = self._load_file(label_file_path, expand_dims)
            # load label images transformer
            self.label_transform = transformer.label_transform()
            if mask_file_path is not None:
                self.mask = self._load_file(mask_file_path, expand_dims)
                self.mask_transform = transformer.weight_transform()
        else:
            self.label = None
            self.label_transform = None
            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                if len(self.raw.shape) == 4:
                    pad_width = ((0, 0), (z, z), (y, y), (x, x))
                self.raw = np.pad(self.raw, pad_width=pad_width, mode='reflect')    
        slice_builder = get_slice_builder(self.raw, self.label, self.mask, slice_builder_config)
        
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.mask_slices = slice_builder.weight_slices
        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])
        mask_patch_transformed = None
        if self.mask is not None:
            mask_idx = self.mask_slices[idx]
            mask_patch_transformed = self.mask_transform(self.mask[mask_idx])
        if self.phase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.label[label_idx])
            if mask_patch_transformed is not None:
                return raw_patch_transformed, label_patch_transformed, mask_patch_transformed
            return raw_patch_transformed, label_patch_transformed
        # img = self.raw[idx]
        # mask = self.mask[idx]
        # if self.phase != 'test':
        #     label = self.label[idx]
        #     # print(np.max(mask))
        #     return self.raw_transform(img), self.label_transform(label), -1, self.mask_transform(mask)
        # else:
        #     return self.raw_transform(img), self.mask_transform(weight), self.paths[idx]

    def __len__(self):
        return self.patch_count

    # @classmethod
    # def prediction_collate(cls, batch):
    #     return img_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        raw_dir = dataset_config.get('raw_dir', 'images')
        label_dir = dataset_config.get('label_dir', '1st_manual')
        mask_dir = dataset_config.get('mask_dir', None)

        # mirror padding conf
        mirror_padding = dataset_config.get('mirror_padding', None)
        expand_dims = dataset_config.get('expand_dims', True)

        raw_file_paths = cls.traverse_paths(file_paths, raw_dir)
        label_file_paths = cls.traverse_paths(file_paths, label_dir)
        mask_file_paths = [None for _ in range(len(raw_file_paths))]
        if mask_dir is not None:
            mask_file_paths = cls.traverse_paths(file_paths, mask_dir)
        
        assert len(raw_file_paths) == len(label_file_paths), 'the number of raw images is not equal to the number of label images.'
        assert len(raw_file_paths) == len(mask_file_paths), 'the number of raw images is not equal to the number of mask images.'
        datasets = []        
        for idx in range(len(raw_file_paths)):
            raw_file_path = raw_file_paths[idx]
            label_file_path = label_file_paths[idx]
            mask_file_path = mask_file_paths[idx]

            try:
                logger.info(f'Loading {phase} set from: {raw_file_path}...')
                dataset = cls(raw_file_path=raw_file_path,
                              label_file_path =label_file_path,
                              mask_file_path=mask_file_path,
                              phase=phase, 
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=mirror_padding,
                              expand_dims = True)
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {raw_file_path}', exc_info=True)
        return datasets
    # we don't know the if the sizes of all images are the same, therefore we create a dataset for each sample. 
    @staticmethod
    def _load_file(file_path, expand_dims):
        img = np.asarray(imageio.imread(file_path))
        if expand_dims:
            dims = img.ndim
            img = np.expand_dims(img, axis=0)
            if dims == 3:
                img = np.transpose(img, (3, 0, 1, 2))
        return img

    @staticmethod
    def traverse_paths(roots, subfolder):
        assert isinstance(roots, list)
        results = []
        for root_dir in roots:
            target_dir = root_dir + '/' + subfolder
            if os.path.isdir(target_dir):
                # if file path is a directory take all NII files in that directory
                iters = [sorted(glob.glob(os.path.join(target_dir, '*'))) ]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                print('root_dir/subfolder should be a directory: root_dir/subfolder/img')
        return results