import sys
import json

import time

import torch

import torchvision
import matplotlib.pyplot as plt

from fMRI import DatasetContainer
from fMRI import DatasetLoader

from fMRI.preprocessing import *
from fMRI.models.reconstruction import UNet
from fMRI.trainer import Trainer

a = DatasetContainer()
a.fastMRI(path='testing/', datasetname='fastMRI', dataset_type='training')


transforms = torchvision.transforms.Compose([ComplexNumpyToTensor(),
                                             KspaceToImage(complex_absolute=True, coil_rss=True),
                                             CropImage()])


loader = DatasetLoader(datasetcontainer=a, transforms=transforms)

model = UNet(n_channels=1, n_classes=1)

path = 'fMRI/config/models/UNet/template.json'

with open(path, 'r') as inifile:
    config = json.load(inifile)

torch_loader = torch.utils.data.DataLoader(loader, batch_size=1, num_workers=2)

for i in torch_loader:
    print(i.shape)
    time.sleep(2)
    print('----------')


trainer = Trainer(
    model=model,
    loss_function=None,
    metric_ftns=None,
    optimizer=None,
    config=config,
    data_loader=None,
    valid_data_loader=None,
    lr_scheduler=None,
    seed=None
    )


