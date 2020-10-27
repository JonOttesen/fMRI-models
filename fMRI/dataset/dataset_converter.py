from pathlib import Path
from copy import deepcopy
from typing import Union, Tuple
import h5py
import pickle
import base64

from tqdm import tqdm
import numpy as np
import torch
import torchvision

import nibabel as nib

from ..dataset import (
    DatasetContainer,
    DatasetEntry,
    DatasetInfo,
    DatasetLoader,
    )

class DatasetConverter(object):

    def __init__(self,
                 container: DatasetContainer,
                 train_transform: torchvision.transforms.Compose,
                 truth_transform: torchvision.transforms.Compose,
                 shape: Tuple[int] = (2, 320, 320),
                 open_func: callable = lambda entry: entry.open()['kspace'],
                 ):

        self.container = deepcopy(container)
        self.train_transform = train_transform
        self.truth_transform = truth_transform

        self.shape = shape
        self.open_func = open_func

    def to_array(self, entry):

        data = self.open_func(entry)
        channels, col, row = self.shape
        volume = np.zeros((len(data), 2, channels, col, row), dtype=np.float32)

        for i, img in enumerate(data):
            train_img = self.train_transform(img)
            truth_img = self.truth_transform(img)

            if isinstance(train_img, torch.Tensor):
                volume[i, 0] = train_img.numpy()
            else:
                volume[i, 0] = train_img

            if isinstance(truth_img, torch.Tensor):
                volume[i, 1] = truth_img.numpy()
            else:
                volume[i, 1] = truth_img

        return volume

    def to_hdf5(self, save_folder: Union[str, Path]):

        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        new_container = deepcopy(self.container)
        for entry, new_entry in tqdm(zip(self.container, new_container), total=len(self.container)):
            volume = self.to_array(entry=entry)

            filename = Path(entry.image_path).stem  # Not really necessary with stem here, however it is more explicit
            save_path = save_folder / Path(filename + '.h5')

            hf = h5py.File(save_path, 'w')
            hf.create_dataset('kspace', data=volume)
            new_entry.image_path = save_path
            new_entry.shape = volume.shape

        return new_container

    def to_nii(self, save_folder: Union[str, Path], affine: np.ndarray = None):

        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        affine = affine if affine is not None else np.eye(4)

        new_container = deepcopy(self.container)
        for entry, new_entry in tqdm(zip(self.container, new_container), total=len(self.container)):
            volume = self.to_array(entry=entry)

            filename = Path(entry.image_path).stem  # Not really necessary with stem here, however it is more explicit
            save_path = save_folder / Path(filename + '.nii')

            nii_image = nib.Nifti1Image(volume, affine=affine)
            nib.save(nii_image, str(save_path))

            new_entry.image_path = save_path
            new_entry.shape = volume.shape

        return new_container

    def to_compressed(self, save_folder: Union[str, Path]):
        """
        Not tested, but the function the code is derived from works
        """
        try:
            import pyzstd
        except:
            raise ImportError('Package pyzstd not installed')

        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        new_container = deepcopy(self.container)
        for entry, new_entry in tqdm(zip(self.container, new_container), total=len(self.container)):
            volume = self.to_array(entry=entry)

            filename = Path(entry.image_path).stem  # Not really necessary with stem here, however it is more explicit
            save_path = save_folder / Path(filename + '.h5')

            n = len(volume)
            hf = h5py.File(save_path, 'w')
            dt = h5py.string_dtype(encoding='utf-8')
            dset = hf.create_dataset('kspace', shape=(n,), dtype=dt)

            for i, slic in enumerate(volume):
                k_bytes = pickle.dumps(slic)
                comp = pyzstd.compress(k_bytes, level_or_option=10)
                dset[i] = base64.b64encode(comp).decode('utf-8')

            new_entry.image_path = save_path

        return new_container



