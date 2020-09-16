from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 1):
    x = torch.arange(start=0, end=size, step=1) - size // 2
    gauss = torch.exp(-x**2/(2*sigma**2)).unsqueeze(0)
    gauss = (gauss.T @ gauss).unsqueeze(0)
    gauss /= gauss.sum()
    gauss = gauss.unsqueeze(0)
    gauss = torch.cat([gauss]*channels, dim=0)

    return gauss


class SSIM(nn.Module):
    """
    Original Paper: http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
    """

    def __init__(self,
                 size: int = 7,
                 sigma: float = 1.5/11*7,
                 channels: int = 1,
                 k1: float = 0.01,
                 k2: float = 0.03):

        super(SSIM, self).__init__()
        self.size = size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.gaussian_window = torch.nn.Parameter(gaussian_window(size=self.size, sigma=self.sigma, channels=channels))

    def apply_conv2d(self, X: torch.Tensor, pad: int = 0):
        channels = X.shape[1]
        return F.conv2d(input=X,
                        weight=self.gaussian_window,
                        stride=1,
                        padding=pad,
                        groups=channels)

    def forward(self,
                X: torch.Tensor,
                Y: torch.Tensor,
                data_range: Union[int, float] = 1):

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        channels = X.shape[1]

        pad = self.size // 2

        mux = self.apply_conv2d(X, pad=pad)
        muy = self.apply_conv2d(Y, pad=pad)

        mux_sq = mux.pow(2)
        muy_sq = muy.pow(2)
        muxy = mux * muy

        sigmax_sq = self.apply_conv2d(X * X, pad=pad) - mux_sq
        sigmay_sq = self.apply_conv2d(Y * Y, pad=pad) - muy_sq
        sigmaxy = self.apply_conv2d(X * Y, pad=pad) - muxy

        ssim_map = ((2*muxy + C1)*(2*sigmaxy + C2)/
                   ((mux_sq + muy_sq + C1)*(sigmax_sq + sigmay_sq + C2)))

        return 1 - ssim_map.mean()


class MS_SSIM(object):
    """
    Supposedly more robust version of the SSIM
    Code Source: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    Possible later implementation
    """

if __name__=='__main__':
    a = torch.rand((1, 1, 256, 256))
    b = torch.rand((1, 1, 256, 256))
    loss = SSIM(channels=3)
    print(loss(a, b))
