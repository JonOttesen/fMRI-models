from pathlib import Path

import numpy as np

import nibabel as nib

from .dataset import (
    DatasetContainer,
    DatasetEntry,
    DatasetInfo,
    DatasetLoader,
    )

def save_as_nii(save_folder,
                dataloader,
                datacontainer,
                train_transform,
                truth_transform,
                ):

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    for entry, (data, target) in zip(datacontainer, dataloader):

        volumes = np.zeros((len(data), 1, 320, 320, 2, 2), dtype=np.float32)

        for i, (sub, full) in enumerate(zip(data, target)):
            volumes[i, :, :, :, 0] = train_transform(sub).numpy()
            volumes[i, :, :, :, 1] = truth_transform(full).numpy()

        filename = Path(entry.image_path).stem
        save_path = save_folder / Path(filename + '.nii')

        nii_image = nib.Nifti1Image(volumes, affine=np.eye(4))
        nib.save(nii_image, str(save_path))

        entry.image_path = str(save_path)

    return datacontainer
