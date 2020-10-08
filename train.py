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

from fMRI.models import MultiLoss, MultiMetric
from fMRI.models.reconstruction import UNet
from fMRI.trainer import Trainer
from fMRI.masks import KspaceMask
from fMRI.config import ConfigReader

from fMRI.preprocessing import (
    Normalization,
    ApplyMaskColumn,
    KspaceToImage,
    ComplexNumpyToTensor,
    CropImage
    )

from fMRI.models.reconstruction.losses import SSIM


train = DatasetContainer()
train.fastMRI(path='train_test_files/', datasetname='fastMRI', dataset_type='training')

valid = DatasetContainer()
valid.fastMRI(path='valid_test_files/', datasetname='fastMRI', dataset_type='validation')

train.to_json(path='./docs/train_files.json')
valid.to_json(path='./docs/valid_files.json')

"""
img = train[1]
img = img.open()
# print(img['mask'][()])
kspace = img['kspace'][5, 10]

plt.imshow(np.log(np.abs(kspace) + 1e-9))
plt.show()
"""

mask_generator = KspaceMask(acceleration=4)
mask = mask_generator.mask_linearly_spaced(lines=320, seed=42)

train_transforms = torchvision.transforms.Compose([
    ComplexNumpyToTensor(),
    # ApplyMaskColumn(mask=mask),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage(size=(320, 320))])

truth_transforms = torchvision.transforms.Compose([
    ComplexNumpyToTensor(),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage()])


training_loader = DatasetLoader(
    datasetcontainer=train,
    train_transforms=train_transforms,
    truth_transforms=truth_transforms
    )

validation_loader = DatasetLoader(
    datasetcontainer=valid,
    train_transforms=train_transforms,
    truth_transforms=truth_transforms
    )


loss = [(1, torch.nn.L1Loss()), (1, SSIM())]
loss = MultiLoss(losses=loss)

train_loader = torch.utils.data.DataLoader(dataset=training_loader, num_workers=1, batch_size=2, shuffle=False)
valid_loader = torch.utils.data.DataLoader(dataset=validation_loader, num_workers=1, batch_size=2, shuffle=False)

metrics = {'SSIM': SSIM(), 'MSE': torch.nn.MSELoss(), 'L1': torch.nn.L1Loss()}
metrics = MultiMetric(metrics=metrics)

path = 'fMRI/config/models/UNet/template.json'

model = UNet(n_channels=1, n_classes=1)

config = ConfigReader(config=path)

optimizer = config.optimizer(model_params=model.parameters())
lr_scheduler = config.lr_scheduler(optimizer=optimizer)


trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    optimizer=optimizer,
    config=config.configs(),
    data_loader=train_loader,
    valid_data_loader=train_loader,
    lr_scheduler=lr_scheduler,
    seed=42
    )

trainer.train()


