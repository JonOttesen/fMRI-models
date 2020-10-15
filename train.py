import torch

import torchvision

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

# "save_dir": "/mnt/CRAI-NAS/all/jona/fMRI/UNet",

# train = DatasetContainer()
# train.fastMRI(path='/mnt/CRAI-NAS/all/jingpeng/data/fastmri/brain/multicoil_train', datasetname='fastMRI', dataset_type='training')

# valid = DatasetContainer()
# valid.fastMRI(path='/mnt/CRAI-NAS/all/jingpeng/data/fastmri/brain/multicoil_val', datasetname='fastMRI', dataset_type='validation')

# test = DatasetContainer()
# test.fastMRI(path='/mnt/CRAI-NAS/all/jingpeng/data/fastmri/brain/multicoil_test', datasetname='fastMRI', dataset_type='test')

train = DatasetContainer.from_json(path='./docs/train_files.json')
valid = DatasetContainer.from_json(path='./docs/valid_files.json')
# test = DatasetContainer.from_json(path='./docs/test_files.json')

# train = DatasetContainer()
# train.fastMRI(path='/home/jon/Documents/CRAI/fMRI/train_test_files', datasetname='fastMRI', dataset_type='training')


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
    truth_transforms=truth_transforms
    )

validation_loader = DatasetLoader(
    datasetcontainer=valid,
    train_transforms=train_transforms,
    truth_transforms=truth_transforms
    )


loss = [(1, torch.nn.L1Loss()), (1, SSIM())]
loss = MultiLoss(losses=loss)


metrics = {'SSIM': SSIM(), 'MSE': torch.nn.MSELoss(), 'L1': torch.nn.L1Loss()}
metrics = MultiMetric(metrics=metrics)

path = './fMRI/config/models/UNet/template.json'

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

trainer.resume_checkpoint(
    resume_model=
    resume_metric=,
    )

trainer.train()


