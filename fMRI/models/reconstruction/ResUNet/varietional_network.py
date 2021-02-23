import torch
import torch.nn.functional as F
import torch.nn as nn

import torchvision

from fMRI.models.blocks import (
    BasicBlock,
    BasicUpBlock,
    Bottleneck,
    Swish,
    MemoryEfficientSwish,
    )
from fMRI.models.reconstruction.ResUNet.repeated_bifpn_resunet import ResUNet

from fMRI.preprocessing import (
    ApplyMaskColumn,
    KspaceToImage,
    PadKspace,
    ComplexNumpyToTensor,
    CropImage,
    ZNormalization,
    ComplexAbsolute,
    RSS,
    PhaseImage,
    ImageToKspace,
    )

class VarNet(nn.Module):

    def __init__(self,
                 cascades: int,
                 n: int = 32,
                 n_repeats: int = 1,
                 BiFPN_layers: int = 4,
                 ratio: float = 1./8,
                 bias: bool = True,
                 ):
        super().__init__()
        self.sense = SensitivityModule(n=n, n_repeats=n_repeats, ratio=ratio, bias=bias, BiFPN=BiFPN_layers)

        self.image_cascades = nn.ModuleList(
            [VarNetBlock(n=n,
                         n_repeats=n_repeats,
                         BiFPN=BiFPN_layers,
                         ratio=ratio,
                         bias=bias) for i in range(cascades)]
        )
        self.kspace_cascades = nn.ModuleList(
            [VarNetKBlock(n=n,
                         n_repeats=n_repeats,
                         BiFPN=BiFPN_layers,
                         ratio=ratio,
                         bias=bias) for i in range(cascades)]
        )

        self.magnitude = torchvision.transforms.Compose([
            KspaceToImage(norm='forward'),
            ComplexAbsolute(),
            RSS(),
            ZNormalization(dim=0),
            ])

        self.complex_imgs = torchvision.transforms.Compose([
            KspaceToImage(norm='forward'),
            ZNormalization(dim=1),
            ])

    def calculate_sense(self, sense_maps: torch.Tensor, output: torch.Tensor):
        coils, _, h, w = sense_maps.shape
        coil_images = torch.zeros(1, coils, h, w).type(torch.complex64)

        # Mapping the image over to the real and imag parts of the coils
        for i, coil in enumerate(sense_maps):
            coil_images[0, i].real = coil[0]*output[0, 0]
            coil_images[0, i].imag = coil[1]*output[0, 0]

        return coil_images

    def calculate_mask(self, tensor):
        tensor = torch.absolute(tensor)
        tensor = torch.sum(tensor, dim=1)  # sum over coils
        tensor = torch.sum(tensor, dim=1)  # sum over height
        tensor[tensor != 0] = 1
        return tensor.type(torch.bool)[0]


    def forward(self, tensor):
        kspace = tensor
        mask = self.calculate_mask(tensor)
        magnitude = self.magnitude(tensor.squeeze(0)).unsqueeze(0)
        complex_img = self.complex_imgs(tensor.squeeze(0)).unsqueeze(0)
        sense_maps = self.sense(magnitude, complex_img)

        for i_cascade, k_cascade in zip(self.image_cascades, self.kspace_cascades):
            x = i_cascade(magnitude)
            coil_images = self.calculate_sense(sense_maps, output=x)
            tensor, means, stds = k_cascade(tensor=coil_images, mask=mask, kspace=kspace)
            exit()

        return


class SensitivityModule(nn.Module):

    def __init__(self, n: int, n_repeats: int, BiFPN: int, ratio: int, bias: bool):
        super().__init__()
        self.model = ResUNet(n_channels=3, n_classes=2, n=n, n_repeats=n_repeats, ratio=ratio, bias=bias, activation=nn.ReLU())
        self.norm = nn.InstanceNorm2d(num_features=3)

    def forward(self, magnitude, coil_images):
        b, c, h, w = coil_images.shape
        sense_maps = torch.zeros(size=(c, 2, h, w))

        for i in range(c):
            x = torch.zeros(b, 3, h, w)
            x[:, 0] = coil_images[:, i].real
            x[:, 1] = coil_images[:, i].imag
            x[:, 2] = magnitude[:, 0]
            x = self.norm(x)
            sense_maps[i, 0] = x[0, 0]
            sense_maps[i, 1] = x[0, 1]

        return sense_maps



class VarNetBlock(nn.Module):

    def __init__(self, n: int, n_repeats: int, BiFPN: int, ratio: int, bias: bool):
        super().__init__()
        self.model = ResUNet(n_channels=1, n_classes=1, n=n, n_repeats=n_repeats, ratio=ratio, bias=bias, activation=nn.ReLU())


    def forward(self,
                tensor: torch.Tensor,
                ):
        """
        Args:
            tensor (torch.Tensor): undersampled image of shape (batch, c, h, w)
        """
        output = self.model(tensor)
        return output


class VarNetKBlock(nn.Module):

    def __init__(self, n: int, n_repeats: int, BiFPN: int, ratio: int, bias: bool):
        super().__init__()
        self.model = ResUNet(n_channels=2, n_classes=2, n=n, n_repeats=n_repeats, ratio=ratio, bias=bias, activation=nn.ReLU())
        self.image_to_kspace = ImageToKspace()
        self.norm = nn.InstanceNorm2d(num_features=2)

    def forward(self,
                tensor: torch.Tensor,
                mask: torch.Tensor,
                kspace: torch.Tensor,
                ):
        """
        Args:
            tensor (torch.Tensor): undersampled image of shape (batch, coil, h, w)
            mask (torch.Tensor): binary mask for which lines are present i.e [False, False, True, False, False, True,....]
        """
        tensor = self.image_to_kspace(tensor.squeeze(0))

        # kspace = (kspace - kspace[:, :, :, mask == 1].mean())/kspace[:, :, :, mask == 1].std()
        # tensor = (tensor - tensor[:, :, :, mask == 1].mean())/tensor[:, :, :, mask == 1].std()

        tensor[:, :, mask == 1] = kspace[:, :, :, mask == 1]
        coils, h, w = tensor.shape
        means = torch.zeros(coils, 2)
        stds = torch.zeros(coils, 2)

        # Correct k-space, in k-space
        for i in range(coils):
            x = torch.zeros(1, 2, h, w)
            x[0, 0] = tensor[i].real
            x[0, 1] = tensor[i].imag

            means[i, 0] = x[0, 0].mean()
            means[i, 1] = x[0, 1].mean()

            stds[i, 0] = x[0, 0].std()
            stds[i, 1] = x[0, 1].std()

            x = self.norm(x)
            x = self.model(x)
            print(x.shape)

            tensor[i].real = x[0, 0]
            tensor[i].imag = x[0, 1]

        return tensor.unsqueeze(0), means, stds
