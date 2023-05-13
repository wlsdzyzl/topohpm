import glob
import os
from itertools import chain

import SimpleITK as sitk
import numpy as np

from toposeg.augment import transforms
from toposeg.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from toposeg.unet3d.utils import get_logger
logger = get_logger('NIIDataset')

class AbstractNIIDataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the NII files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, raw_file_path, label_file_path, 
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 truncation = None,
                 weight_file_path=None,
                 global_normalization=True):
        """
        :param file_path: path to NII file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param raw_internal_path (str or list): NII internal path to the raw dataset
        :param label_internal_path (str or list): NII internal path to the label dataset
        :param weight_file_path (str or list): NII internal path to the per pixel weights
        """
        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase
        self.file_path = raw_file_path
        self.raw = self.create_nii_file(raw_file_path)
        if truncation is not None:
            print(truncation)
            self.raw[self.raw > truncation[1]] = truncation[1]
            self.raw[self.raw < truncation[0]] = truncation[0] 
        if global_normalization:
            stats = calculate_stats(self.raw)
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase == 'test':
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None
            self.weight_map = None
            self.skeletal_points = None
            self.radius = None
            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                if self.raw.ndim == 4:
                    channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in self.raw]
                    self.raw = np.stack(channels)
                else:
                    self.raw = np.pad(self.raw, pad_width=pad_width, mode='reflect')
        else:
            # When will this happen?
            if label_file_path is None: 
                self.label = None
                self.weight_map = None
                self.skeletal_points = None
                self.radius = None
            else:
                # create label/weight transform only in train/val phase
                self.label_transform = self.transformer.label_transform()
                self.label = self.create_nii_file(label_file_path)
                
                # self.label = 1 - (self.label > 0).astype(float)
                if weight_file_path is not None:
                    # look for the weight map in the raw file
                    self.weight_map = self.create_nii_file(weight_file_path)
                    self.weight_transform = self.transformer.weight_transform()
                else:
                    self.weight_map = None
                self._check_volume_sizes(self.raw, self.label)

                    # print(self.skeletal_points)

        # build slice indices for raw and label data sets
        print('loader: ', self.raw.shape)
        slice_builder = get_slice_builder(self.raw, self.label, self.weight_map, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices
        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])

        if self.phase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.label[label_idx])
            if self.weight_map is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.weight_map[weight_idx])
                return raw_patch_transformed, label_patch_transformed, self.idx, weight_patch_transformed, 
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_nii_file(file_path):
        raise NotImplementedError

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        print(raw.shape, label.shape)
        assert _volume_shape(raw) == _volume_shape(label), 'import imageio'  
        
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        raw_dir = dataset_config.get('raw_dir', 'raw')
        label_dir = dataset_config.get('label_dir', None)
        weight_dir = dataset_config.get('weight_dir', None)
        
        raw_file_paths = cls.traverse_nii_paths(file_paths, raw_dir)
        if label_dir is not None:
            label_file_paths = cls.traverse_nii_paths(file_paths, label_dir)
        else:
            label_file_paths = [None for _ in raw_file_paths]
        if weight_dir is not None:
            weight_file_paths = cls.traverse_nii_paths(file_paths, weight_dir)
        else:
            weight_file_paths = [None for _ in raw_file_paths]


        datasets = []
        for idx in range(len(raw_file_paths)):
            raw_file_path = raw_file_paths[idx]
            label_file_path = label_file_paths[idx]
            weight_file_path = weight_file_paths[idx]
            print(label_file_path, raw_file_path)
            try:
                logger.info(f'Loading {phase} set from: {raw_file_path}...')
                dataset = cls(raw_file_path=raw_file_path,
                              label_file_path =label_file_path,
                              weight_file_path=weight_file_path,
                              phase=phase, 
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              truncation = dataset_config.get('truncation', None),
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
                # only one sample
                # break
            except Exception:
                logger.error(f'Skipping {phase} set: {raw_file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_nii_paths(roots, subfolder):
        assert isinstance(roots, list)
        results = []
        for root_dir in roots:
            target_dir = root_dir + '/' + subfolder
            if os.path.isdir(target_dir):
                # if file path is a directory take all NII files in that directory
                iters = [sorted(glob.glob(os.path.join(target_dir, '*.nii.gz'))) ]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                print('root_dir/subfolder should be a directory: root_dir/subfolder/data.nii')
        return results




class StandardNIIDataset(AbstractNIIDataset):
    """
    Implementation of the NII dataset which loads the data from all of the NII files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, raw_file_path, label_file_path, phase, slice_builder_config, transformer_config, mirror_padding=(16, 32, 32),
                 weight_file_path=None, 
                 truncation = None,
                 global_normalization=True):
        super().__init__(raw_file_path=raw_file_path,
                         label_file_path=label_file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         weight_file_path=weight_file_path,
                         truncation = truncation,
                         global_normalization=global_normalization)

    @staticmethod
    def create_nii_file(file_path):
        image = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(image)