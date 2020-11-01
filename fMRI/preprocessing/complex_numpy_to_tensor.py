import torch
import numpy as np

from .fastmri import transforms as T

class ComplexNumpyToTensor(object):
    """
    ***torchvision.Transforms compatible***

    Converts a numpy complex array to a torch tensor where the real and
    imaginary parts are stacked along the last dimension
    """

    def __call__(self, tensor: np.ndarray):
        """
        Args:
            tensor (np.ndarry): Array with shape (batch, coils, rows, columns)
        Returns:
            torch.Tensor: The torch.Tensor version of the complex numpy array
                          with shape (batch, coils, rows, columns) with the
                          last dim being the real and complex part

        """

        return torch.from_numpy(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'
