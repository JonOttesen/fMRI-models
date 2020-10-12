from typing import Union

import torch
import numpy as np


class Normalization(object):
    """
    ***torchvision.Transforms compatible***

    Normalizes the image in the interval [0, 1] by (x-xmin)/(xmax - xmin)
    along the specified dimension
    """
    def __init__(self, dim: int = 1):
        """
        Args:
            dim (int): the dimension for the normalization i.e for a batch of images(size=(batch, i, j, k))
                       dim=0 would normalize with respect to the entire batch whereas dim=1 normalizes
                       each individual image. For shape (i, j, k) dim=0 normalizes across the i'th dimension
        """
        self.dim = dim

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor image to be normalized along the given dim

        Returns:
            torch.Tensor: Normalized Tensor image.

        """

        return normalize(img=tensor, dim=self.dim)

    def __repr__(self):
        return self.__class__.__name__ + '(dim={0})'.format(self.dim)



def normalize(img: Union[torch.Tensor, np.ndarray],
              dim: int = 0,
              minimum: Union[float, int] = None,
              maximum: Union[float, int] = None):
    """
    Normalizes the image in the interval [0, 1] by (x-xmin)/(xmax - xmin)
    along the specified dimension
    Args:
        img: (torch.Tensor, np.ndarray), the image/array/something which is to be normalized
        dim: (int), the dimension for the normalization i.e for a batch of images(size=(batch, i, j, k))
            dim=0 would normalize with respect to the entire batch whereas dim=1 normalizes
            each individual image
    returns:
        (torch.Tensor, np.ndarray), the normalized version of the input along the specified dimension
    """
    norm_dim = tuple(range(dim, img.ndim))

    if minimum and maximum:
        return (img - minimum)/(maximum - minimum)

    if isinstance(img, torch.Tensor):
        img = img.numpy()

    minimum = img.min(axis=norm_dim, keepdims=True)
    maximum = img.max(axis=norm_dim, keepdims=True)

    return torch.from_numpy((img - minimum)/(maximum - minimum))


if __name__=='__main__':
    import time
    a = torch.randn(size=(10, 40, 800))
    start_time = time.time()
    for i in range(int(1e4)):
        img = normalize(a, dim=0)
    print('Finished', time.time() - start_time)

    start_time = time.time()
    for i in range(int(1e4)):
        img_np = normalize(a, dim=0)
    print('Finished', time.time() - start_time)
