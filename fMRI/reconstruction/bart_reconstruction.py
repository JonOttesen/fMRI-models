from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..logger import get_logger

logger = get_logger(name=__name__)

try:
    from bart import bart
except ModuleNotFoundError:
    logger.warning('Could not import bart module, '
        'please install Bart before trying to use it for reconstruction')


class BartReconstruction:

    @classmethod
    def _prepeare_input(self, tensor: Union[torch.Tensor, np.ndarray]):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        tensor = np.transpose(tensor, axes=(1, 2, 0))
        return tensor

    @classmethod
    def coil_sensitivity(cls,
                         tensor: Union[np.ndarray, torch.Tensor],
                         m: int = 1,
                         ):
        """
        Args:
            tensor (np.ndarray, torch.Tensor): Input image to be reconstructed.
                shape: (c, h, w) as expected from pytorch inputs
            m (int): Total number of coil_sensitivity maps for each coil
                default is 1.
        returns:
            (np.ndarray): Coil sensitivities from the input kspace
                output shape (h, w, z, c, m) as given by the bart toolbox
                where m is the number of maps as specified in the input and
                z is the third dimension in the scan
        """
        tensor = cls._prepeare_input(tensor)
        return bart(1, 'ecalib -m{}'.format(m), tensor[:,:,None,:])

    @classmethod
    def paralell_imageing(cls,
                          tensor: Union[np.ndarray, torch.Tensor],
                           m: int = 1,
                           i: int = 50,
                           r: float = 0.01,
                           ):
        """
        Args:
            tensor (np.ndarray, torch.Tensor): Input image to be reconstructed.
                shape: (c, h, w) as expected from pytorch inputs
            m (int): Total number of coil_sensitivity maps for each coil
                default=1.
            i (int): number of iterations, default=50
            r (float): regularization parameter, default=0.01
        returns:
            (np.ndarray): the reconstructed image using parallel imaging
                output shape (h, w)
        """
        sens = cls.coil_sensitivity(tensor=tensor, m=m)
        tensor = cls._prepeare_input(tensor)
        recon = bart(1, 'pics -l2 -i{0} -r{1}'.format(i, r), tensor[:,:,None,:], sens)
        return recon

    @classmethod
    def compressed_sensing(cls,
                           tensor: Union[np.ndarray, torch.Tensor],
                           m: int = 1,
                           i: int = 50,
                           r: float = 0.01,
                           ):
        """
        Args:
            tensor (np.ndarray, torch.Tensor): Input image to be reconstructed.
                shape: (c, h, w) as expected from pytorch inputs
            m (int): Total number of coil_sensitivity maps for each coil
                default=1.
            i (int): number of iterations, default=50
            r (float): regularization parameter, default=0.01
        returns:
            (np.ndarray): the reconstructed image using compressed sensing
                output shape (h, w)
        """
        sens = cls.coil_sensitivity(tensor=tensor, m=m)
        tensor = cls._prepeare_input(tensor)
        recon = bart(1, 'pics -l1 -i{0} -r{1}'.format(i, r), tensor[:,:,None,:], sens)
        return recon
