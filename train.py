import sys
import json

import time

from pathlib import Path

import torch
import numpy as np

import torchvision
import matplotlib.pyplot as plt

from fMRI import DatasetContainer
from fMRI import DatasetLoader
from fMRI import DatasetInfo

from fMRI.preprocessing import *
from fMRI.models import MultiLoss
from fMRI.models.reconstruction import UNet
from fMRI.trainer import Trainer
from fMRI.masks import KspaceMask
from fMRI.config import ConfigReader

from fMRI.models.reconstruction.losses import SSIM


a = DatasetContainer()
b = a.fastMRI(path='testing/', datasetname='fastMRI', dataset_type='training')

img = b[1]
img = img.open()
# print(img['mask'][()])
kspace = img['kspace'][5, 10]

plt.imshow(np.log(np.abs(kspace) + 1e-9))
plt.show()


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

loss = [(1, torch.nn.L1Loss()), (1, SSIM())]
loss = MultiLoss(losses=loss)

load = torch.utils.data.DataLoader(dataset=loader, num_workers=1, batch_size=2)

metrics = {'SSIM': SSIM(), 'MSE': torch.nn.MSELoss()}

model = UNet(n_channels=1, n_classes=1)

path = 'fMRI/config/models/UNet/template.json'

config = ConfigReader(config=path)

optimizer = config.optimizer(model_params=model.parameters())
lr_scheduler = config.lr_scheduler(optimizer=optimizer)


trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    optimizer=optimizer,
    config=config.configs(),
    data_loader=load,
    valid_data_loader=None,
    lr_scheduler=lr_scheduler,
    seed=42
    )

trainer.train()


