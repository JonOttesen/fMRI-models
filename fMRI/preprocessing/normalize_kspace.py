import torch

import torchvision

from fMRI.masks import KspaceMask

class NormalizeKspace(object):
    """
    ***torchvision.Transforms compatible***

    Downsamples the FOV by fourier than cropping than inverse fouirer
    """
    def __init__(self, center_fraction: float = 0.08, return_max: bool = False):
        """
        Args:
            dim (int): the dimension for downsampling, 1 for height and 2 for width
            size (int): the length of k-space along the dim direction
        """
        self.center_fraction = center_fraction
        self.return_max = return_max

    def __call__(self, tensor: torch.Tensor):
        """
        Calculates the phase images
        Args:
            tensor (torch.Tensor): Complex Tensor of the image data with shape
                                   (coils, rows, columns, ...) i.e ndim >= 1
        Returns:
            torch.Tensor: The real phase images with equal shape

        """
        x = tensor.clone()
        lines = tensor.shape[2]
        cent = tensor.shape[2] // 2
        frac = int(lines*self.center_fraction // 2)
        mxx = torch.max(torch.abs(x[:, :, cent-frac:cent+frac]))
        tensor = tensor/mxx

        if self.return_max:
            return tensor, mxx

        return tensor


    def __repr__(self):
        return self.__class__.__name__ + '(center_fraction={0})'.format(self.center_fraction)

