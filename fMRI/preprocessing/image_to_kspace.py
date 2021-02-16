import torch

from .fastmri import math
import matplotlib.pyplot as plt

class ImageToKspace(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given image data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 norm: str = 'backward',
                 ):
        """
        Args:
            norm (str): normalization method used in the fft transform,
                see doc for torch.fft.fft for possible args
        """
        self.norm = norm

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the image data with shape
                                   (coils, rows, columns) i.e ndim=3
        Returns:
            torch.Tensor: The fft transformed image to k-space with shape
                          (channels, rows, columns)

        """
        tensor_dtype = tensor.dtype
        tensor = tensor.type(torch.complex128)
        if isinstance(tensor, torch.Tensor):
            data = math.fftshift(tensor, dim=(-2, -1))
            data = torch.fft.fftn(data, dim=(-2, -1), norm=self.norm)
            data = math.ifftshift(data, dim=(-2, -1))
            return data.type(tensor_dtype)
        else:
            raise TypeError('tensor need to be torch.Tensor or np.ndarray')

    def __repr__(self):
        return self.__class__.__name__ + '(norm={0})'.format(self.norm)
