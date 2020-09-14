from typing import Union
import h5py

from pathlib import Path


class DatasetEntry(object):

    def __init__(self,
                 image_path: Union[str, Path] = None,
                 datasetname: str = None,
                 dataset_type: str = None,
                 sequence_type: str = None,
                 field_strength: float = None,
                 pre_contrast: bool = None,
                 post_contrast: bool = None,
                 multicoil: bool = None):

        if isinstance(image_path, (Path, str)):
            self.image_path = str(image_path)
            if not Path(image_path).is_file():
                print('The path: ', str(image_path))
                print('Is not an existing file, are you sure this is the correct path?')
        else:
            self.image_path = image_path

        self.datasetname = datasetname
        self.dataset_type = dataset_type

        self.sequence_type = sequence_type
        self.field_strength = field_strength

        if not isinstance(pre_contrast, bool) and pre_contrast is not None:
            raise TypeError('The variable pre_contrast ', pre_contrast, ' need to be boolean')

        if not isinstance(multicoil, bool) and multicoil is not None:
            raise TypeError('The variable multicoil ', pre_contrast, ' need to be boolean')

        self.pre_contrast = pre_contrast
        self.post_contrast = post_contrast
        self.multicoil = multicoil

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def open(self, open_func=None):

        if open_func is not None:
            image = open_func(self.image_path)
        else:
            suffix = Path(self.image_path).suffix
            if suffix == '.h5':
                image = self.open_hdf5(self.image_path)
            elif suffix in ['.nii', '.gz']:
                image = self.open_nifti(self.image_path)

        return image

    def open_hdf5(self, image_path):
        return h5py.File(image_path)

    def open_nifti(self, image_path):
        return NotImplementedError


    def keys(self):
        return self.to_dict().keys()

    def to_dict(self) -> dict:
        """
        returns:
            dict format of this class
        """
        return {'image_path': self.image_path,
                'datasetname': self.datasetname,
                'dataset_type': self.dataset_type,
                'sequence_type': self.sequence_type,
                'field_strength': self.field_strength,
                'pre_contrast': self.pre_contrast,
                'post_contrast': self.post_contrast,
                'multicoil': self.multicoil}

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.image_path = in_dict['image_path']
            self.datasetname = in_dict['datasetname']
            self.dataset_type = in_dict['dataset_type']
            self.sequence_type = in_dict['sequence_type']
            self.field_strength = in_dict['field_strength']
            self.pre_contrast = in_dict['pre_contrast']
            self.post_contrast = in_dict['post_contrast']
            self.multicoil = in_dict['multicoil']

        return self