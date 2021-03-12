import torch

import torchvision

from .kspace_to_image import KspaceToImage
from .image_to_kspace import ImageToKspace

class DownsampleFOV(object):
    """
    ***torchvision.Transforms compatible***

    Downsamples the FOV by fourier than cropping than inverse fouirer
    """
    def __init__(self, k_size: int = 320, i_size: int = 320):
        """
        Args:
            dim (int): the dimension for downsampling, 1 for height and 2 for width
            size (int): the length of k-space along the dim direction
        """
        self.k_size = k_size
        self.i_size = i_size

    def __call__(self, tensor: torch.Tensor):
        """
        Calculates the phase images
        Args:
            tensor (torch.Tensor): Complex Tensor of the image data with shape
                                   (coils, rows, columns, ...) i.e ndim >= 1
        Returns:
            torch.Tensor: The real phase images with equal shape

        """

        fft = KspaceToImage(norm='ortho')
        ifft = ImageToKspace(norm='ortho')
        i_crop = torchvision.transforms.CenterCrop(self.i_size)
        k_crop = torchvision.transforms.CenterCrop(self.k_size)

        tensor = fft(tensor)
        tensor = i_crop(tensor)
        tensor = ifft(tensor)

        if not self.k_size == self.i_size:
            tensor = k_crop(tensor)

        return tensor


    def __repr__(self):
        return self.__class__.__name__ + '(k_size={0}, i_size={1})'.format(self.k_size, self.i_size)

