import torch

import torchvision

from fMRI import DatasetContainer
from fMRI import DatasetLoader
from fMRI import DatasetInfo

from fMRI.models import MultiLoss, MultiMetric
from fMRI.metrics import (
    NMSE,
    PSNR,
    )

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

from fMRI.models.reconstruction.losses import SSIM, FSSIMLoss

# Inspect folders
train = DatasetContainer()
train.fastMRI(path='folder/where/fastmri/training_data/is/stored', datasetname='fastMRI', dataset_type='training')

valid = DatasetContainer()
valid.fastMRI(path='folder/where/fastmri/validation_data/is/stored', datasetname='fastMRI', dataset_type='validation')

test = DatasetContainer()
test.fastMRI(path='folder/where/fastmri/test_data/is/stored', datasetname='fastMRI', dataset_type='test')

# Write to files
train.to_json(path='./docs/train_files.json')
valid.to_json(path='./docs/valid_files.json')
test.to_json(path='./docs/test_files.json')

# Open files, to skip the inspection stage each time you train
train = DatasetContainer.from_json(path='./docs/train_files.json')
valid = DatasetContainer.from_json(path='./docs/valid_files.json')
test = DatasetContainer.from_json(path='./docs/test_files.json')

# Making mask instance
mask = KspaceMask(acceleration=4, seed=42)

# Crating the transforms the kspace training data must go through
train_transforms = torchvision.transforms.Compose([
    PadKspace(shape=(320, 320)),
    ComplexNumpyToTensor(),
    ApplyMaskColumn(mask=mask),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage(size=(320, 320)),
    ZNormalization(dim=0),
    ])

# Crating the transforms the kspace validation data must go through, note no apply mask
truth_transforms = torchvision.transforms.Compose([
    PadKspace(shape=(320, 320)),
    ComplexNumpyToTensor(),
    KspaceToImage(complex_absolute=True, coil_rss=True),
    CropImage(size=(320, 320)),
    ZNormalization(dim=0),
    ])


# Create DatasetLoaders which open the files and is pytorch dataloader_compat
training_loader = DatasetLoader(
    datasetcontainer=train,
    train_transforms=train_transforms,
    truth_transforms=truth_transforms,
    dataloader_compat=True,
    )

validation_loader = DatasetLoader(
    datasetcontainer=valid,
    train_transforms=train_transforms,
    truth_transforms=truth_transforms,
    dataloader_compat=True,
    )

# Create loss functions, multiple is stored in a list as tuple
loss = [(1, torch.nn.L1Loss()), (1, SSIM())]
loss = MultiLoss(losses=loss)

# Metrics used in validation
metrics = {
    'SSIM': SSIM(),
    'FSSIM': FSSIMLoss(),
    'PSNR': PSNR(),
    'NMSE': NMSE(),
    'MSE': torch.nn.MSELoss(),
    'L1': torch.nn.L1Loss()}

metrics = MultiMetric(metrics=metrics)

path = 'path/to/config.json'

model = UNet(n_channels=1, n_classes=1)

config = ConfigReader(config=path)

train_loader = torch.utils.data.DataLoader(dataset=training_loader,
                                           num_workers=config.num_workers,
                                           batch_size=config.batch_size,
                                           shuffle=config.shuffle)


valid_loader = torch.utils.data.DataLoader(dataset=validation_loader,
                                           num_workers=config.num_workers,
                                           batch_size=config.batch_size,
                                           shuffle=config.shuffle)

# Optim and scheduler from config
optimizer = config.optimizer(model_params=model.parameters())
lr_scheduler = config.lr_scheduler(optimizer=optimizer)


trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    optimizer=optimizer,
    config=config.configs(),
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    lr_scheduler=lr_scheduler,
    seed=42
    )


exit()

trainer.resume_checkpoint(
    resume_model='/resume/path/2020-10-18/epoch_15/checkpoint-epoch15.pth',
    resume_metric='/resume_path/2020-10-18/epoch_15/statistics.json',
    )

trainer.train()


