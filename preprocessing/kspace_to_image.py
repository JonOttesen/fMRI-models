import torch

import fastmri

class KspaceToImage(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given k-space data to the corresponding image using Fourier transforms
    """
    def __init__(self, complex_absolute: bool = True):
        """
        Args:
            complex_absolute (bool): Whether to do take the rss between the real
                                     and complex number, default is True
        """
        self.complex_absolute = complex_absolute

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with shape
                                   (batch, coils, rows, columns, complex) i.e ndim=5
        Returns:
            torch.Tensor: The ifft transformed k-space to image with shape
                          (batch, rows, columns) or (batch, rows, columns, complex)
                          if the complex_absolute is not calculated

        """
        a = fastmri.ifft2c(tensor)
        if self.complex_absolute:
            b = fastmri.complex_abs(a)
        else:
            b = a
        c = fastmri.rss(b, dim=1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(complex_absolute={0})'.format(self.complex_absolute)
