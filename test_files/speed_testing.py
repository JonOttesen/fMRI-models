import sys
import json

import time

import logging
import h5py
import blosc
import base64

from tqdm import tqdm

from pathlib import Path

import torch
import numpy as np

import torchvision
import matplotlib.pyplot as plt

from fMRI import DatasetContainer
from fMRI import DatasetLoader
from fMRI import DatasetInfo

from fMRI.models import MultiLoss, MultiMetric
from fMRI.models.reconstruction import UNet
from fMRI.trainer import Trainer
from fMRI.masks import KspaceMask
from fMRI.config import ConfigReader

from fMRI.preprocessing import (
    Normalization,
    ApplyMaskColumn,
    KspaceToImage,
    PadKspace,
    ComplexNumpyToTensor,
    CropImage,
    ZNormalization,
    )

from fMRI.models.reconstruction.losses import SSIM

train = DatasetContainer()
train.fastMRI(path='/home/jona/Documents/CRAI/Compressed_testing', datasetname='fastMRI', dataset_type='training')

# train = DatasetContainer.from_json(path='./docs/train_files.json')


img = train[0]
img = img.open()
kspace = img['kspace']


hf = h5py.File('compression_test.h5')
hf.create_dataset('kspace', data=kspace, compression='blosc', compression_opts=9)

hf.close()

exit()





mask = KspaceMask(acceleration=4, seed=42)



train_transforms = torchvision.transforms.Compose([
    PadKspace(shape=(320, 320)),
    ComplexNumpyToTensor(),
    ApplyMaskColumn(mask=mask),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage(size=(320, 320)),
    ZNormalization(dim=0),
    ])

truth_transforms = torchvision.transforms.Compose([
    PadKspace(shape=(320, 320)),
    ComplexNumpyToTensor(),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage(size=(320, 320)),
    ZNormalization(dim=0),
    ])


training_loader = DatasetLoader(
    datasetcontainer=train,
    train_transforms=train_transforms,
    truth_transforms=train_transforms
    )

train_loader = torch.utils.data.DataLoader(dataset=training_loader,
                                           num_workers=4,
                                           batch_size=4,
                                           shuffle=False)

test_size = 500
print('Start time check')
start_time = time.time()
counter = 0
for j in train_loader:
    counter += 1
    if counter >= test_size:
        break

print('DataLoader file loading time W PP: {}'.format(time.time() - start_time))


training_loader = DatasetLoader(
    datasetcontainer=train,
    # train_transforms=train_transforms,
    # truth_transforms=train_transforms
    )

train_loader = torch.utils.data.DataLoader(dataset=training_loader,
                                           num_workers=4,
                                           batch_size=4,
                                           shuffle=False)

test_size = 500
print('Start time check')
start_time = time.time()
counter = 0
for j in train_loader:
    counter += 1
    if counter >= test_size:
        break

print('DataLoader file loading time WO PP: {}'.format(time.time() - start_time))



# Home computer relative speeds
# Time WO any transforms:                   8.7 seconds
# Time with only ComplexNumpyToTensor:      18.3 seconds
# Time with CNTT and KspaceToImage:         79.8 seconds
# Time with everything:                     99.1 seconds


# Improvements:
