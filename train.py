import sys
import json

import time

import torch

import torchvision
import matplotlib.pyplot as plt

from fMRI import DatasetContainer
from fMRI import DatasetLoader
from fMRI import DatasetInfo

from fMRI.preprocessing import *
from fMRI.models.reconstruction import UNet
from fMRI.trainer import Trainer
from fMRI.masks import KspaceMask

from fMRI.models.reconstruction.losses import SSIM

a = DatasetContainer()
b = a.fastMRI(path='testing/', datasetname='fastMRI', dataset_type='training')

mask_generator = KspaceMask(acceleration=4)
mask = mask_generator.mask_linearly_spaced(lines=320, seed=42)

train_transforms = torchvision.transforms.Compose([
    ComplexNumpyToTensor(),
    ApplyMaskColumn(mask=mask),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage(size=(320, 320))]
    )

truth_transforms = torchvision.transforms.Compose([
    ComplexNumpyToTensor(),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage()]
    )


loader = DatasetLoader(
    datasetcontainer=a,
    train_transforms=train_transforms,
    truth_transforms=truth_transforms
    )

loss = lambda x, y: SSIM()(x, y) + torch.nn.L1Loss()(x, y)

load = torch.utils.data.DataLoader(dataset=loader, num_workers=2, batch_size=2)

for (i, j) in load:
    print(i.shape)
    print(j.shape)
    print('------------')

model = UNet(n_channels=1, n_classes=1)

path = 'fMRI/config/models/UNet/template.json'

with open(path, 'r') as inifile:
    config = json.load(inifile)

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=None,
    optimizer=None,
    config=config,
    data_loader=loader,
    valid_data_loader=None,
    lr_scheduler=None,
    seed=42
    )


