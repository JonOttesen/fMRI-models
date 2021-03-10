"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fMRI.models.reconstruction.VarNet.fastmri as fastmri


from fMRI.models.reconstruction.ResUNet.repeated_bifpn_resunet import ResUNet

from fMRI.preprocessing import (
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

from .varnet import NormResUnet, VarNetBlock, SensitivityModel

class PhaseVarNet(nn.Module):
    """
    A full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 4,
        var_n: int = 16,
        sense_n: int = 8,
        n_repeats: int = 1,
        BiFPN_layers: int = 0,
        ratio: float = 1./8,
        bias: bool = True,
        ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            n_channels = 2,
            n_classes = 2,
            n = sense_n,
            n_repeats = n_repeats,
            BiFPN_layers = BiFPN_layers,
            ratio = ratio,
            bias = bias,
            )

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormResUnet(
                n_channels = 2,
                n_classes = 2,
                n = var_n,
                n_repeats = n_repeats,
                BiFPN_layers = BiFPN_layers,
                ratio = ratio,
                bias = bias,
            )) for _ in range(num_cascades)])

    def calculate_mask(self, masked_kspace: torch.Tensor):
        x = masked_kspace[:, :, :, :, 0]**2 + masked_kspace[:, :, :, :, 1]**2
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sum(x, dim=2, keepdim=True).unsqueeze(-1)
        x = x != 0
        return x


    def forward(self, masked_kspace: torch.Tensor) -> torch.Tensor:
        mask = self.calculate_mask(masked_kspace)
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)


        x = fastmri.ifft2c(kspace_pred)
        x = fastmri.rss(torch.atan2(x[:, :, :, :, 1], x[:, :, :, :, 0]), dim=1)
        return x

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1).unsqueeze(1)
