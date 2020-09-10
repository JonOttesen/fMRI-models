import json
from pathlib import Path
from typing import Union, List
from copy import deepcopy
import numpy as np

from .DatasetEntry import DatasetEntry
from .DatasetInfo import DatasetInfo


class DatasetContainer(object):

    def __init__(self,
                 info: List[DatasetInfo] = list(),
                 entries: List[DatasetEntry] = list()):

        self.entries = entries
        self.info = info

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)

    def __delitem__(self, index):
        del self.entries[index]

    def __str__(self):
        return str(self.to_dict())

    def info_dict(self):
        info_dict = dict()
        for inf in self.info:
            info_dict[inf.datasetname] = inf.to_dict()

        return info_dict

    def copy(self):
        return deepcopy(self.entries)

    def add_info(self, info: DatasetInfo):
        self.info.append(info)

    def add_entry(self, entry: DatasetEntry):
        self.entries.append(entry)

    def keys(self):
        return self.to_dict().keys()

    def to_dict(self):
        container_dict = dict()
        container_dict['info'] = [inf.to_dict() for inf in self.info]
        container_dict['entries'] = [entry.to_dict() for entry in self.entries]
        return container_dict

    def from_dict(self, in_dict):
        self.info.append(DatasetInfo().from_dict(inf) for inf in in_dict['info'])
        self.entries.append(DatasetEntry().from_dict(entry) for entry in in_dict['entries'])

    def to_json(self, path: Union[str, Path]):
        path = Path(path)
        suffix = path.suffix
        if suffix != '.json':
            raise NameError('The path must have suffix .json not, ', suffix)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(self.to_dict(), outfile, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        with open(path) as json_file:
            data = json.load(json_file)
        DatasetContainer().from_dict(data)

    def fastMRI(self,
                path: Union[str, Path],
                datasetname: str,
                dataset_type: str,
                source: str = 'fastMRI',
                dataset_description: str = 'Data for fastMRI challenge',
                multicoil=True):

        """
        Fills up the container using the folder containing the fastMRI data
        Args:
            path: (str, pathlib.Path), path to the fastMRI data
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError('path argument is {}, expected type is pathlib.Path or str'.format(type(path)))

        path = path.absolute()
        files = list(path.glob('*.h5'))

        info = DatasetInfo(
            datasetname=datasetname,
            dataset_type=dataset_type,
            source=source,
            dataset_description=dataset_description
            )

        self.add_info(info=info)

        for file in files:
            filename = file.name

            entry = DatasetEntry(
                image_path=file,
                datasetname=datasetname,
                dataset_type=dataset_type,
                multicoil=multicoil
                )

            # Pre or post contrast or None if neither
            if 'PRE' in filename or 'pre' in filename or 'Pre' in filename:
                entry.pre_contrast = True
            elif 'POST' in filename or 'post' in filename or 'Post' in filename:
                entry.post_contrast = True

            if 'T2' in filename or 't2' in filename:
                entry.sequence_type = 'T2'
            elif 'T1' in filename or 't1' in filename:
                entry.sequence_type = 'T1'
            elif 'FLAIR' in filename or 'flair' in filename or 'Flair' in filename:
                entry.sequence_type = 'FLAIR'

            self.add_entry(entry=entry)

        return self


