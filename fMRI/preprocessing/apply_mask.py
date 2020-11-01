import torch

class ApplyMaskColumn(object):
    """
    ***torchvision.Transforms compatible***

    Applies the given mask to the k-space data and sets the non
    specified columns to zero
    """

    def __init__(self, mask: torch.Tensor):
        """
        Args:
            mask (torch.Tensor): The mask used in under-sampling the given k-space data,
                                 assumes shape: (number_of_columns_in_kspace)
        """
        self.mask = mask

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with
                                   shape (coil, rows, columns) or (rows, columns)
        Returns:
            torch.Tensor: K-space tensor with same shape and applied mask on columns

        """
        shape = tensor.shape
        mask = self.mask(shape[-1])
        if len(shape) == 3:
            tensor[:, :, mask != True] = 0
        else:
            tensor[:, mask != True] = 0

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mask={0})'.format(self.mask)
