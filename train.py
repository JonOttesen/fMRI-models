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

a = DatasetContainer()
b = a.fastMRI(path='testing/', datasetname='fastMRI', dataset_type='training')


transforms = torchvision.transforms.Compose([ComplexNumpyToTensor(),
                                             KspaceToImage(complex_absolute=True, coil_rss=True),
                                             CropImage()])


loader = DatasetLoader(datasetcontainer=a, transforms=transforms)

for i in loader:
    pass

model = UNet(n_channels=1, n_classes=1)

path = 'fMRI/config/models/UNet/template.json'

with open(path, 'r') as inifile:
    config = json.load(inifile)

trainer = Trainer(
    model=model,
    loss_function=None,
    metric_ftns=None,
    optimizer=None,
    config=config,
    data_loader=loader,
    valid_data_loader=None,
    lr_scheduler=None,
    seed=42
    )


