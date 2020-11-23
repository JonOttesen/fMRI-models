from typing import Union

import torch
import torchvision.transforms.functional as F
import numpy as np


class ZNormalization(object):
    """
    ***torchvision.Transforms compatible***

    ZNormalizes the image by (x-mean(x))/(std(x))
    along the specified dimension
    """
    def __init__(self, dim: int = 0, inplace: bool = False):
        """
        Args:
            dim (int): the dimension for the normalization. For shape (i, j, k, ...) dim=0
            normalizes takes the mean and std of the i, j, k, ... 'th dimension. For dim=1
            the mean and std is taken for the j, k, ... 'th dimension
        """
        self.dim = dim
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor image to be normalized along the given dim

        Returns:
            torch.Tensor: Normalized Tensor image.

        """
        norm_dim = tuple(range(self.dim, tensor.ndim))
        norm = F.normalize(tensor=tensor,
                           mean=tensor.mean(axis=norm_dim),
                           std=tensor.std(axis=norm_dim),
                           inplace=self.inplace,
                           )

        return norm

    def __repr__(self):
        return self.__class__.__name__ + '(dim={0}, inplace={1})'.format(self.dim, self.inplace)


if __name__=='__main__':
    img = torch.randn(size=(10, 40, 800))
    norm = ZNormalization(dim=1)
    a = norm(tensor=img)
    print(torch.std(a))
