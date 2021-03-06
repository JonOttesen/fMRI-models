import torch

from .fastmri import math

import matplotlib.pyplot as plt

class KspaceToImage(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given k-space data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 norm: str = 'backward',
                 ):
        """
        Args:
            norm (str): normalization method used in the ifft transform,
                see doc for torch.fft.ifft for possible args
        """
        self.norm = norm

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with shape
                                   (coils, rows, columns) i.e ndim=3
        Returns:
            torch.Tensor: The ifft transformed k-space to image with shape
                          (channels, rows, columns)
        """
        if isinstance(tensor, torch.Tensor):
            data = math.ifftshift(tensor, dim=(-2, -1))
            data = torch.fft.ifftn(data, dim=(-2, -1), norm=self.norm)
            data = math.fftshift(data, dim=(-2, -1))
            return data
        else:
            raise TypeError('tensor need to be torch.Tensor or np.ndarray')

    def __repr__(self):
        return self.__class__.__name__ + '(norm={0})'.format(self.norm)
