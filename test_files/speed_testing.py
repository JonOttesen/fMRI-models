import sys
import json

import time

import logging
import h5py
import base64
import pickle

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
train.fastMRI(path='/home/jon/Documents/CRAI/fMRI/train_test_files', datasetname='fastMRI', dataset_type='training')

# train = DatasetContainer.from_json(path='./docs/train_files.json')


img = train[0]
img = img.open()
kspace = img['kspace'][()]

import pyzstd
import sys
import time

start = time.time()

n = len(kspace)
hf = h5py.File('test_files/compression_test.h5')
dt = h5py.string_dtype(encoding='utf-8')
dset = hf.create_dataset('kspace', shape=(n,), dtype=dt)

for i, slic in enumerate(kspace):
    k_bytes = pickle.dumps(slic)
    comp = pyzstd.compress(k_bytes, level_or_option=-1)
    dset[i] = base64.b64encode(comp).decode('utf-8')

uncomp = pyzstd.decompress(comp)
array = pickle.loads(uncomp)

k_bytes = pickle.dumps(kspace)
comp = pyzstd.compress(k_bytes, level_or_option=-1)
a = base64.b64encode(comp).decode('utf-8')

start = time.time()
for i in range(100):
    comp = base64.b64decode(a.encode('utf-8'))
    uncomp = pyzstd.decompress(comp)
    array = pickle.loads(uncomp)


print('Time for 100 files: ', time.time() - start)
print('Og object: ', sys.getsizeof(k_bytes)/1024**2)
print('Comp object: ', sys.getsizeof(comp)/1024**2)
print('Uncomp object: ', sys.getsizeof(uncomp)/1024**2)

print(array.shape)
print(array is kspace)


print((array == kspace).all())

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
