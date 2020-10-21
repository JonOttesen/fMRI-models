import pickle
from typing import Union
from pathlib import Path

import base64
import pyzstd
import h5py

from .datasetcontainer import DatasetContainer

def save_compressed(container: DatasetContainer,
                    save_folder: Union[str, Path]):

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    import sys
    for entry in container:
        filename = Path(entry.image_path).name
        save_path = save_folder / Path(filename)

        kspace = entry.open()['kspace']

        n = len(kspace)
        hf = h5py.File(save_path)
        dt = h5py.string_dtype(encoding='utf-8')
        dset = hf.create_dataset('kspace', shape=(n,), dtype=dt)
        size = 0
        for i, slic in enumerate(kspace):
            k_bytes = pickle.dumps(slic)
            comp = pyzstd.compress(k_bytes, level_or_option=10)
            dset[i] = base64.b64encode(comp).decode('utf-8')
            size += sys.getsizeof(comp)/1024**2
        print(size)

        entry.image_path = save_path

    return container

