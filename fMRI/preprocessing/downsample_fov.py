import torch

import torchvision

from .kspace_to_image import KspaceToImage
from .image_to_kspace import ImageToKspace

class DownsampleFOV(object):
    """
    ***torchvision.Transforms compatible***

    Downsamples the FOV by fourier than cropping than inverse fouirer
    """
    def __init__(self, dim: int = 2, crop_size: int = 320, size: int = 64):
        """
        Args:
            dim (int): the dimension for downsampling, 1 for height and 2 for width
            size (int): the length of k-space along the dim direction
        """
        assert dim in [1, 2], "dim need to be either 1 or 2"
        self.dim = dim
        self.size = size
        self.crop_size = crop_size

    def __call__(self, tensor: torch.Tensor):
        """
        Calculates the phase images
        Args:
            tensor (torch.Tensor): Complex Tensor of the image data with shape
                                   (coils, rows, columns, ...) i.e ndim >= 1
        Returns:
            torch.Tensor: The real phase images with equal shape

        """
        height, width = tensor.shape[-2], tensor.shape[-1]
        cropping = (self.crop_size, width) if self.dim==1 else (height, self.crop_size)
        fft = KspaceToImage(norm='forward')
        ifft = ImageToKspace(norm='forward')
        crop = torchvision.transforms.CenterCrop(cropping)
        k_crop = torchvision.transforms.CenterCrop(self.size)

        tensor = fft(tensor)
        tensor = crop(tensor)
        tensor = ifft(tensor)
        tensor = k_crop(tensor)
        return tensor


    def __repr__(self):
        return self.__class__.__name__ + '(dim={0}, size={1})'.format(self.dim, self.size)

