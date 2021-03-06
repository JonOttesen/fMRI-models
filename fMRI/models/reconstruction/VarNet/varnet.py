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
    NormalizeKspace,
    ComplexSplit,
    )

class NormResUnet(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = 2,
        n: int = 16,
        n_repeats: int = 1,
        BiFPN_layers: int = 0,
        ratio: float = 1./8,
        bias: bool = True,
        ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.model = ResUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            n=n,
            n_repeats=n_repeats,
            BiFPN_layers=BiFPN_layers,
            ratio=ratio,
            bias=bias,
            )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.model(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = 2,
        n: int = 8,
        n_repeats: int = 1,
        BiFPN_layers: int = 0,
        ratio: float = 1./8,
        bias: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.model = NormResUnet(
            n_channels = n_channels,
            n_classes = n_classes,
            n = n,
            n_repeats = n_repeats,
            BiFPN_layers = BiFPN_layers,
            ratio = ratio,
            bias = bias,
            )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)


    def mask_center(self, x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
        """
        Initializes a mask with the center filled in.
        Args:
            mask_from: Part of center to start filling.
            mask_to: Part of center to end filling.
        Returns:
            A mask with the center filled.
        """
        mask = torch.zeros_like(x)
        mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

        return mask


    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # get low frequency line locations and mask them out
        cent = mask.shape[-2] // 2

        cent = mask.shape[-2] // 2
        left = torch.nonzero(mask.squeeze()[:cent] == 0)[-1]
        right = torch.nonzero(mask.squeeze()[cent:] == 0)[0] + cent
        num_low_freqs = right - left
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2

        x = self.mask_center(masked_kspace, pad, pad + num_low_freqs)

        # convert to image space
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)

        # estimate sensitivities
        x = self.model(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)


        return x


class VarNet(nn.Module):
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
        inference: bool = False,
        center_fraction: bool = 0.08,
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
        self.inference = inference
        self.center_fraction = center_fraction

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

    def normalize_kspace(self, masked_kspace: torch.Tensor) -> torch.Tensor:
        masked_kspace, maxx = NormalizeKspace(center_fraction=self.center_fraction, return_max=True)(masked_kspace)
        self.maxx = maxx
        return masked_kspace

    def forward(self, masked_kspace: torch.Tensor) -> torch.Tensor:
        if self.inference:
            assert masked_kspace.dtype in [torch.complex32, torch.complex64, torch.complex128],\
            "in inference, input must be of complex tensor"
            masked_kspace = self.normalize_kspace(masked_kspace)
            masked_kspace = ComplexSplit()(masked_kspace)


        mask = self.calculate_mask(masked_kspace)
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        if self.inference:
            kspace_pred = kspace_pred*self.maxx

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1).unsqueeze(1)



class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        ) -> torch.Tensor:

        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term
