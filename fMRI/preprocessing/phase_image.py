import torch

class PhaseImage(object):
    """
    ***torchvision.Transforms compatible***

    Calculates the Phase image
    """

    def __call__(self, tensor: torch.Tensor):
        """
        Calculates the phase images
        Args:
            tensor (torch.Tensor): Complex Tensor of the image data with shape
                                   (coils, rows, columns, ...) i.e ndim >= 1
        Returns:
            torch.Tensor: The real phase images with equal shape

        """
        tensor = torch.arctan(tensor.imag/tensor.real)

        return tensor


    def __repr__(self):
        return self.__class__.__name__
