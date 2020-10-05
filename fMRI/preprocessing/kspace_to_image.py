import torch

from .fastmri import math
from .fastmri.coil_combine import rss

class KspaceToImage(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given k-space data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 complex_absolute: bool = True,
                 coil_rss: bool = True):
        """
        Args:
            complex_absolute (bool): Whether to do take the rss between the real
                                     and complex number, default is True
        """
        self.complex_absolute = complex_absolute
        self.coil_rss = coil_rss

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

        a = math.ifft2c(tensor)
        if self.complex_absolute:
            b = math.complex_abs(a)
        else:
            b = a

        if self.coil_rss:
            return torch.unsqueeze(rss(b, dim=0), 0)  # Add a channels dimension
        else:
            return b  # Use the coils as channels

    def __repr__(self):
        return self.__class__.__name__ + '(complex_absolute={0})'.format(self.complex_absolute)
