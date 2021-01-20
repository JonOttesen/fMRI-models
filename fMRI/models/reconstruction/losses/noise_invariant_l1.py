from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseInvariantL1(nn.Module):
    """
    Just testing out some noise invarience for L1 loss
    """

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss(reduction='none')

    def __repr__(self):
        return self.__class__.__name__

    def forward(self,
                X: torch.Tensor,
                Y: torch.Tensor,
                sigma: float = None,
                ):
        """
        Args:
            X (torch.Tensor): Prediction
            Y (torch.Tensor): Ground truth
            data_range (float): Difference between maximum and minimum value
        """
        if sigma is None:
            sigma = Y[:, :, :50, :50].cpu().detach().std(axis=(1, 2, 3))
            print(sigma)

        l1 = self.l1(X, Y)
        factor = 1 - torch.exp(-(l1 - sigma)**2/(2*sigma**2))
        loss = l1*factor

        return torch.mean(loss)

if __name__=='__main__':
    a = torch.rand((3, 100, 100))
    b = torch.rand((3, 100, 100)) + 2

    loss = NoiseInvariantL1()
    std = a.std()
    loss(a, b, sigma=None)
