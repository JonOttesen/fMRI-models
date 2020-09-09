from pathlib import Path
import h5py

import torchvision

from .DatasetContainer import DatasetContainer
from .DatasetEntry import DatasetEntry
from .DatasetInfo import DatasetInfo

class DatasetLoader(object):

    def __init__(self,
                 datasetcontainer: DatasetContainer,
                 transforms: torchvision.transforms,
                 open_func: callable = None,
                 hdf5_key: str = 'kspace'):

        self.datasetcontainer = datasetcontainer
        self.transforms = transforms
        self.open_func = open_func
        self.hdf5_key = hdf5_key

    def __len__(self):
        return len(self.datasetcontainer)

    def __getitem__(self, index):
        entry = self.datasetcontainer[index]
        image_path = entry.image_path

        if self.open_func is not None:
            image = self.open_func(image_path)
        else:
            suffix = Path(image_path).suffix
            if suffix == '.h5':
                image = self.open_hdf5(image_path=image_path)
            elif suffix in ['.nii', '.gz']:
                image = self.open_nifti(image_path=image_path)

        return self.transforms(image)

    def open_hdf5(self, image_path):
        hf = h5py.File(image_path)
        volume = hf[self.hdf5_key][:]
        return volume

    def open_nifti(self, image_path):

        return NotImplementedError


