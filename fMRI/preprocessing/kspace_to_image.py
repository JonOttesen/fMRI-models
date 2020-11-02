import torch

from .fastmri import math

class KspaceToImage(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given k-space data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 norm: str = 'backward',
                 fftw: bool = False,
                 ):
        """
        Args:
            norm (str): normalization method used in the ifft transform,
                see doc for torch.fft.ifft for possible args
        """
        self.norm = norm

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
        if isinstance(tensor, torch.Tensor):
            data = math.ifftshift(tensor, dim=(-2, -1))
            data = torch.fft.ifftn(data, dim=(-2, -1), norm=self.norm)
            data = math.fftshift(data, dim=(-2, -1))
            return data
        else:
            raise TypeError('tensor need to be torch.Tensor or np.ndarray with fftw enabled')
        """
        elif isinstance(tensor, np.ndarray) and self.fftw:
            return self.fftw_transform(tensor)
        """

    def __repr__(self):
        return self.__class__.__name__ + '(norm={0})'.format(self.norm)

"""
    def fftw_transform(self, tensor):

        shape = tensor.shape
        tensor = np.fft.ifftshift(tensor, axes=(-2, -1))
        # kspace = pyfftw.empty_aligned(shape, dtype='complex64')
        img = pyfftw.empty_aligned(shape, dtype='complex64')

        ifft_object = pyfftw.FFTW(tensor, img, direction='FFTW_BACKWARD', axes=(-2, -1))

        # kspace[:] = tensor

        ifft_img = np.fft.fftshift(ifft_object(), axes=(-2, -1))
        return ifft_img

        shape = tensor.shape
        if not shape in self.transforms.keys():
            kspace = pyfftw.empty_aligned(shape, dtype='complex64')
            ifft2 = pyfftw.builders.ifft2(kspace, overwrite_input=True)
            self.transforms[shape] = (ifft2, kspace)

        ifft2, kspace = self.transforms[shape]
        kspace[:] = tensor
        return ifft2()
"""
