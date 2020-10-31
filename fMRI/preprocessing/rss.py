import torch
import numpy as np

from .fastmri import math
from .fastmri.coil_combine import rss

class RSS(object):
    """
    ***torchvision.Transforms compatible***

    Calculates the RSS (residual sum of squares) in 0'th dimension
    """

    def __call__(self, tensor: torch.Tensor):
        """
        Takes the residual sum of squares on first dimension often the channels
        Args:
            tensor (torch.Tensor): Tensor of the image data with shape
                                   (coils, rows, columns, ..., complex) i.e ndim >= 1
        Returns:
            torch.Tensor: The ifft transformed k-space to image with shape
                          (channels, rows, columns) or
                          (channels, rows, columns, complex)
                          if the complex_absolute is not calculated

        """

        return torch.unsqueeze(rss(tensor, dim=0), 0)  # Add a channels dimension


    def __repr__(self):
        return self.__class__.__name__
