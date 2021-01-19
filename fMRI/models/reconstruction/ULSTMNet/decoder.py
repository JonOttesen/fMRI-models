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

class Decoder(nn.Module):

    def __init__(self,
                 n_classes: int,
                 n: int = 128,
                 n_repeats: int = 1,
                 ratio: float = 1./8,
                 bias: bool = False,
                 ):
        super().__init__()
        self.n_classes = n_classes

        conv2d = get_same_padding_conv2d()
        self.activation = MemoryEfficientSwish()
        norm = nn.InstanceNorm2d

        self.up1 = BasicUpBlock(8*n, 8*n, stride=2, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.bottle_up1 = nn.Sequential(*[Bottleneck(in_channels=16*n,
                                     mid_channels=16*n // 4,
                                     out_channels=16*n,
                                     ratio=ratio,
                                     norm_layer=norm,
                                     activation_func=self.activation,
                                     bias=bias,
                                     ) for i in range(n_repeats)])
        self.up_conv1 = conv2d(16*n, 4*n, kernel_size=3, bias=bias)

        self.up2 = BasicUpBlock(4*n, 4*n, stride=2, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.bottle_up2 = nn.Sequential(*[Bottleneck(in_channels=8*n,
                                     mid_channels=8*n // 4,
                                     out_channels=8*n,
                                     ratio=ratio,
                                     norm_layer=norm,
                                     activation_func=self.activation,
                                     bias=bias,
                                     ) for i in range(n_repeats)])
        self.up_conv2 = conv2d(8*n, 2*n, kernel_size=3, bias=bias)

        self.up3 = BasicUpBlock(2*n, 2*n, stride=2, norm_layer=norm, activation_func=self.activation)
        self.bottle_up3 = nn.Sequential(*[Bottleneck(in_channels=4*n,
                                     mid_channels=4*n // 4,
                                     out_channels=4*n,
                                     ratio=ratio,
                                     norm_layer=norm,
                                     activation_func=self.activation,
                                     bias=bias,
                                     ) for i in range(n_repeats)])
        self.up_conv3 = conv2d(4*n, n, kernel_size=3, bias=bias)

        self.up4 = BasicUpBlock(n, n, stride=2, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.final_bottle = Bottleneck(in_channels=2*n,
                                 mid_channels=n,
                                 out_channels=2*n,
                                 ratio=ratio,
                                 norm_layer=norm,
                                 activation_func=self.activation,
                                 bias=bias,
                                 )
        self.outc = nn.Conv2d(in_channels=2*n, out_channels=1, stride=1, kernel_size=1)


    def forward(self, inputs):

        x1, x2, x3, x4, x = inputs

        x = self.up1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.bottle_up1(x)
        x = self.activation(self.up_conv1(x))

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.bottle_up2(x)
        x = self.activation(self.up_conv2(x))

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.bottle_up3(x)
        x = self.activation(self.up_conv3(x))

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.final_bottle(x)

        return self.outc(x)
