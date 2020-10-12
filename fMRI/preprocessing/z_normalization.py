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
    def __init__(self, dim: int = 1, inplace: bool = False):
        """
        Args:
            dim (int): the dimension for the normalization i.e for a batch of images(size=(batch, i, j, k))
                       dim=0 would normalize with respect to the entire batch whereas dim=1 normalizes
                       each individual image. For shape (i, j, k) dim=0 normalizes across the i'th dimension
            inplace (bool,optional): Bool to make this operation in-place.
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
        tensor = tensor.numpy()
        return F.normalize(tensor=torch.from_numpy(tensor),
                           mean=tensor.mean(axis=norm_dim),
                           std=tensor.std(axis=norm_dim),
                           inplace=self.inplace,
                           )

    def __repr__(self):
        return self.__class__.__name__ + '(dim={0}, inplace={1})'.format(self.dim, self.inplace)


if __name__=='__main__':
    img = torch.randn(size=(10, 40, 800))
    norm = ZNormalization(dim=0)
    print(norm(tensor=img))

