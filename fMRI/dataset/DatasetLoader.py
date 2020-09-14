from pathlib import Path
import h5py

import torch
import torchvision

from .DatasetContainer import DatasetContainer
from .DatasetEntry import DatasetEntry
from .DatasetInfo import DatasetInfo

class DatasetLoader(torch.utils.data.Dataset):

    def __init__(self,
                 datasetcontainer: DatasetContainer,
                 transforms: torchvision.transforms,
                 open_func: callable = None,
                 img_key: str = 'kspace'):

        self.datasetcontainer = datasetcontainer
        self.transforms = transforms
        self.open_func = open_func
        self.img_key = img_key

    def __len__(self):
        return len(self.datasetcontainer)

    def __getitem__(self, index):
        entry = self.datasetcontainer[index]
        suffix = Path(entry.image_path).suffix
        image_object = entry.open(open_func=self.open_func)

        if suffix == '.h5':
            image = image_object[self.img_key][()]
        elif suffix in ['.nii', '.gz']:
            image = image_object[self.img_key][()]

        print(image.shape)
        if self.transforms is not None:
            return self.transforms(image)
        else:
            return image

    def __iter__(self):
        self.current_index = 0
        self.max_length = len(self)
        return self

    def __next__(self):
        if not self.current_index < self.max_length:
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item



