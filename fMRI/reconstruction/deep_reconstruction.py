from typing import Union

import torch

import numpy as np
import matplotlib.pyplot as plt


class DeepReconstruction:

    def __init__(self,
                 model: torch.nn.Module,
                 use_cuda: bool = True):

        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.model = model.to(self.device)

    def reconstruct(self, tensor: Union[torch.Tensor]):
        """
        Args:
            tensor (torch.Tensor): input tensor with size (c, h, w)
        returns:
            the reconstructed image (c, h, w)
        """
        if tensor.ndim == 3:
            tensor = torch.unsqueeze(tensor, 0)

        self.model.eval()

        with torch.no_grad():
            tensor = tensor.to(self.device)
            prediction = self.model(tensor).cpu()

        return prediction

