import torch
import numpy as np
try:
    import pyfftw
except:
    pass

from .fastmri import math
from .fastmri.coil_combine import rss

class KspaceToImage(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given k-space data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 fftw: bool = False):
        """
        Args:
            fftw (bool): Whether to enable fftw instead of pytorch
        """
        self.fftw = fftw
        self.transforms = {}

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with shape
                                   (coils, rows, columns, complex) i.e ndim=4
        Returns:
            torch.Tensor: The ifft transformed k-space to image with shape
                          (channels, rows, columns) or
                          (channels, rows, columns, complex)
                          if the complex_absolute is not calculated

        """
        if isinstance(tensor, torch.Tensor):
            return math.ifft2c(tensor)
        elif isinstance(tensor, np.ndarray) and self.fftw:
            return self.fftw_transform(tensor)
        else:
            raise TypeError('tensor need to be torch.Tensor or np.ndarray with fftw enabled')

    def __repr__(self):
        return self.__class__.__name__ + '(fftw={0})'.format(self.fftw)


    def fftw_transform(self, tensor):
        """
        shape = tensor.shape
        tensor = np.fft.ifftshift(tensor, axes=(-2, -1))
        # kspace = pyfftw.empty_aligned(shape, dtype='complex64')
        img = pyfftw.empty_aligned(shape, dtype='complex64')

        ifft_object = pyfftw.FFTW(tensor, img, direction='FFTW_BACKWARD', axes=(-2, -1))

        # kspace[:] = tensor

        ifft_img = np.fft.fftshift(ifft_object(), axes=(-2, -1))
        return ifft_img
        """
        shape = tensor.shape
        if not shape in self.transforms.keys():
            kspace = pyfftw.empty_aligned(shape, dtype='complex64')
            ifft2 = pyfftw.builders.ifft2(kspace, overwrite_input=True)
            self.transforms[shape] = (ifft2, kspace)

        ifft2, kspace = self.transforms[shape]
        kspace[:] = tensor
        return ifft2()
