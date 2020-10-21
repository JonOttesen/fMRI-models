from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_window(size: int = 11,
                    sigma: float = 1.5,
                    channels: int = 1,
                    gaussian: bool = True):
    if gaussian:
        x = torch.arange(start=0, end=size, step=1) - size // 2
        gauss = torch.exp(-x**2/(2*sigma**2)).unsqueeze(0)
        gauss = (gauss.T @ gauss).unsqueeze(0)
        gauss /= gauss.sum()
        gauss = gauss.unsqueeze(0)
        gauss = torch.cat([gauss]*channels, dim=0)
        return gauss

    return torch.ones((size, size)).unsqueeze(0).unsqueeze(0)/size**2

class SSIM(nn.Module):
    """
    Original Paper: http://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf
    """

    def __init__(self,
                 size: int = 7,
                 sigma: float = 1.5,  # 1.5 is the regular for a size of 11
                 channels: int = 1,
                 k1: float = 0.01,
                 k2: float = 0.03):

        super(SSIM, self).__init__()
        self.size = size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.gaussian_window = torch.nn.Parameter(gaussian_window(size=self.size,
                                                                  sigma=self.sigma,
                                                                  channels=channels))

        NP = size ** 2
        self.cov_norm = NP / (NP - 1)

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
                data_range: float = None):
        """
        Args:
            X (torch.Tensor): Prediction
            Y (torch.Tensor): Ground truth
            data_range (float): Difference between maximum and minimum value
        """
        if not data_range:
            """
            See for more info about data_range
            https://github.com/scikit-image/scikit-image/blob/master/skimage/metrics/_structural_similarity.py#L12-L232
            Assuming the first dimension is the batch_size
            """
            data_range = 2  # Since the std=1 i.e 1 -- 1 = 2
            # batch_size = X.shape[0]
            # X_flattend = X.view(batch_size, -1)
            # data_range = X_flattend.max(dim=1)[0] - X_flattend.min(dim=1)[0]
            # for i in range(len(X.shape) - 1):
                # data_range = torch.unsqueeze(data_range, dim=-1)

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        pad = self.size // 2

        mux = self.apply_conv2d(X, pad=pad)
        muy = self.apply_conv2d(Y, pad=pad)

        mux_sq = mux.pow(2)
        muy_sq = muy.pow(2)
        muxy = mux * muy

        sigmax_sq = (self.apply_conv2d(X * X, pad=pad) - mux_sq)*self.cov_norm
        sigmay_sq = (self.apply_conv2d(Y * Y, pad=pad) - muy_sq)*self.cov_norm
        sigmaxy = (self.apply_conv2d(X * Y, pad=pad) - muxy)*self.cov_norm

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
    torch.manual_seed(42)
    a = torch.rand((15, 1, 150, 206))
    b = torch.rand((15, 1, 150, 206))

    from skimage.metrics import structural_similarity as ssim
    from FSSIM import FSSIMLoss

    loss = SSIM(channels=1)
    g = loss(a, b)
    print(g)

    floss = FSSIMLoss()
    print(floss(a, b, data_range=1))

    ssum = 0
    for i, j in zip(a, b):
        i = i.permute(1, 2, 0)
        j = j.permute(1, 2, 0)
        ssum += 1 - ssim(i.numpy(), j.numpy(), win_size=7, gaussian_weights=True, multichannel=True, use_sample_covariance=False)

    print(ssum/15)
