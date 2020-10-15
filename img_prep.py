import torchvision

from fMRI.save_as_nii import save_as_nii

from fMRI import DatasetContainer
from fMRI import DatasetLoader

from fMRI.trainer import Trainer
from fMRI.masks import KspaceMask

from fMRI.preprocessing import (
    Normalization,
    ApplyMaskColumn,
    KspaceToImage,
    PadKspace,
    ComplexNumpyToTensor,
    CropImage,
    ZNormalization,
    )


# train = DatasetContainer()
# train.fastMRI(path='/mnt/CRAI-NAS/all/jingpeng/data/fastmri/brain/multicoil_train', datasetname='fastMRI', dataset_type='training')

# valid = DatasetContainer()
# valid.fastMRI(path='/mnt/CRAI-NAS/all/jingpeng/data/fastmri/brain/multicoil_val', datasetname='fastMRI', dataset_type='validation')

# test = DatasetContainer()
# test.fastMRI(path='/mnt/CRAI-NAS/all/jingpeng/data/fastmri/brain/multicoil_test', datasetname='fastMRI', dataset_type='test')

# train = DatasetContainer.from_json(path='./docs/train_files.json')
# valid = DatasetContainer.from_json(path='./docs/valid_files.json')
# test = DatasetContainer.from_json(path='./docs/test_files.json')

train = DatasetContainer()
train.fastMRI(path='/home/jon/Documents/CRAI/fMRI/train_test_files', datasetname='fastMRI', dataset_type='training')


mask = KspaceMask(acceleration=4, seed=42)

train_transforms = torchvision.transforms.Compose([
    PadKspace(shape=(320, 320)),
    ComplexNumpyToTensor(),
    ApplyMaskColumn(mask=mask),
    KspaceToImage(complex_absolute=False, coil_rss=True),
    CropImage(size=(320, 320)),
    # ZNormalization(dim=0),
    ])

truth_transforms = torchvision.transforms.Compose([
    PadKspace(shape=(320, 320)),
    ComplexNumpyToTensor(),
    KspaceToImage(complex_absolute=False, coil_rss=True),
    CropImage(size=(320, 320)),
    # ZNormalization(dim=0),
    ])


training_loader = DatasetLoader(
    datasetcontainer=train,
    train_transforms=None,
    truth_transforms=None,
    dataloader_compat=False
    )

save_as_nii(
    save_folder='/home/jon/Documents/CRAI/test_nifti',
    dataloader=training_loader,
    datacontainer=train,
    train_transform=train_transforms,
    truth_transform=truth_transforms,
    )
