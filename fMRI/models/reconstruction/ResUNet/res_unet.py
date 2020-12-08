import torch
import torch.nn.functional as F
import torch.nn as nn

from fMRI.models.blocks import (
    BasicBlock,
    BasicUpBlock,
    Bottleneck,
    Swish,
    MemoryEfficientSwish,
    )
from fMRI.models.blocks.utils import get_same_padding_conv2d

class ResUNet(nn.Module):

    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 n: int = 128,
                 ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        conv2d = get_same_padding_conv2d()
        self.activation = MemoryEfficientSwish()
        norm = nn.InstanceNorm2d

        self.inc = BasicBlock(n_channels, n, norm_layer=norm, activation_func=self.activation)

        self.down1 = BasicBlock(n, n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.conv1 = conv2d(n, 2*n, kernel_size=3)

        self.down2 = BasicBlock(2*n, 2*n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.conv2 = conv2d(2*n, 4*n, kernel_size=3)

        self.down3 = BasicBlock(4*n, 4*n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.conv3 = conv2d(4*n, 8*n, kernel_size=3)

        self.down4 = BasicBlock(8*n, 8*n, stride=2, norm_layer=norm, activation_func=self.activation)


        self.up1 = BasicUpBlock(8*n, 8*n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.up_conv1 = conv2d(16*n, 4*n, kernel_size=3)

        self.up2 = BasicUpBlock(4*n, 4*n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.up_conv2 = conv2d(8*n, 2*n, kernel_size=3)

        self.up3 = BasicUpBlock(2*n, 2*n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.up_conv3 = conv2d(4*n, n, kernel_size=3)

        self.up4 = BasicUpBlock(n, n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.bottle = Bottleneck(in_channels=2*n,
                                 mid_channels=n,
                                 out_channels=2*n,
                                 norm_layer=norm,
                                 activation_func=self.activation,
                                 )
        self.outc = nn.Conv2d(in_channels=2*n, out_channels=1, stride=1, kernel_size=1)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.activation(self.conv1(self.down1(x1)))
        x3 = self.activation(self.conv2(self.down2(x2)))
        x4 = self.activation(self.conv3(self.down3(x3)))
        x = self.down4(x4)


        x = self.up1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.activation(self.up_conv1(x))

        x = self.up2(x)
        x = self.activation(self.up_conv2(torch.cat([x, x3], dim=1)))

        x = self.up3(x)
        x = self.activation(self.up_conv3(torch.cat([x, x2], dim=1)))

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.bottle(x)

        return self.outc(x)
