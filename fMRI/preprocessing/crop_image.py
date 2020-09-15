from typing import Union

import torch


class CropImage(object):
    """
    ***torchvision.Transforms compatible***

    Crops the height and width of the input image to the given size
    """

    def __init__(self, size: Union[tuple, list] = (320, 320)):
        """
        Args:
            crop (tuple, list): The desired shape of the (H, W) of the output image
        """
        self.size = size

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the image after kspace_to_image,
                                   shape either:(channel, height/rows, columns/width,...) i.e ndim == 3
                                   or 4 if the complex absolute is not taken
        Returns:
            torch.Tensor: Cropped torch.Tensor with the size of self.crop

        """
        shape = tensor.shape
        height, width = self.size

        img_h, img_w = shape[1], shape[2]
        return tensor[:, int(img_h/2 - height/2):int(img_h/2 + height/2),\
                      int(img_w/2 - width/2):int(img_w/2 + width/2)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
