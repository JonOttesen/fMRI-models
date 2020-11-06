import torch.nn.functional as F
import torch.nn as nn

from .unet_blocks import DoubleConv, UpConv, DownConv, OutConv


class GELUUNet(nn.Module):

    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 n: int = 128,
                 bilinear: bool = True):
        super(GELUUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, n)
        self.down1 = DownConv(n, 2*n)
        self.down2 = DownConv(2*n, 4*n)
        self.down3 = DownConv(4*n, 8*n)
        self.down4 = DownConv(8*n, 16*n // factor)

        self.up1 = UpConv(16*n, 8*n // factor, bilinear)
        self.up2 = UpConv(8*n, 4*n // factor, bilinear)
        self.up3 = UpConv(4*n, 2*n // factor, bilinear)
        self.up4 = UpConv(2*n, n, bilinear)
        self.outc = OutConv(n, n_classes)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
