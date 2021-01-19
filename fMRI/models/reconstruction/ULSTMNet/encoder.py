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
from fMRI.models.blocks.BiFPN import BiFPN

class Encoder(nn.Module):

    def __init__(self,
                 n_channels: int,
                 n: int = 128,
                 n_repeats: int = 1,
                 BiFPN_layers: int = 4,
                 ratio: float = 1./8,
                 bias: bool = False,
                 ):
        super().__init__()
        self.n_channels = n_channels

        conv2d = get_same_padding_conv2d()
        self.activation = MemoryEfficientSwish()
        norm = nn.InstanceNorm2d

        self.inc = BasicBlock(n_channels, n, norm_layer=norm, activation_func=self.activation, bias=bias)

        self.down1 = BasicBlock(n, n, stride=2, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.conv1 = conv2d(n, 2*n, kernel_size=3, bias=bias)
        self.bottle1 = nn.Sequential(*[Bottleneck(in_channels=2*n,
                                  mid_channels=2*n // 2,
                                  out_channels=2*n,
                                  ratio=ratio,
                                  norm_layer=norm,
                                  activation_func=self.activation,
                                  bias=bias,
                                  ) for i in range(n_repeats)])

        self.down2 = BasicBlock(2*n, 2*n, stride=2, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.conv2 = conv2d(2*n, 4*n, kernel_size=3, bias=bias)
        self.bottle2 = nn.Sequential(*[Bottleneck(in_channels=4*n,
                                  mid_channels=4*n // 4,
                                  out_channels=4*n,
                                  ratio=ratio,
                                  norm_layer=norm,
                                  activation_func=self.activation,
                                  bias=bias,
                                  ) for i in range(n_repeats)])

        self.down3 = BasicBlock(4*n, 4*n, stride=2, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.conv3 = conv2d(4*n, 8*n, kernel_size=3, bias=bias)
        self.bottle3 = nn.Sequential(*[Bottleneck(in_channels=8*n,
                                  mid_channels=8*n // 4,
                                  out_channels=8*n,
                                  ratio=ratio,
                                  norm_layer=norm,
                                  activation_func=self.activation,
                                  bias=bias,
                                  ) for i in range(n_repeats)])

        self.down4 = BasicBlock(8*n, 8*n, stride=2, norm_layer=norm, activation_func=self.activation, bias=bias)

        self.bottle_middle = nn.Sequential(*[Bottleneck(in_channels=8*n,
                                        mid_channels=8*n // 4,
                                        out_channels=8*n,
                                        ratio=ratio,
                                        norm_layer=norm,
                                        activation_func=self.activation,
                                        bias=bias,
                                        ) for i in range(n_repeats)])


        self.bifpns = nn.ModuleList([
            BiFPN(channels=[n, 2*n, 4*n, 8*n, 8*n], layers=5)
            for i in range(BiFPN_layers)])



    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.activation(self.conv1(self.down1(x1)))
        x2 = self.bottle1(x2)

        x3 = self.activation(self.conv2(self.down2(x2)))
        x3 = self.bottle2(x3)

        x4 = self.activation(self.conv3(self.down3(x3)))
        x4 = self.bottle3(x4)

        x = self.down4(x4)
        x = self.bottle_middle(x)

        inputs = [x1, x2, x3, x4, x]
        for bifpn in self.bifpns:
            inputs = bifpn(inputs)
            inputs[0] += x1
            inputs[1] += x2
            inputs[2] += x3
            inputs[3] += x4
            inputs[4] += x
            x1, x2, x3, x4, x = inputs


        return inputs
