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

class ResUNet(nn.Module):

    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 n: int = 128,
                 n_repeats: int = 1,
                 BiFPN_layers: int = 4,
                 ratio: float = 1./8,
                 bias: bool = False,
                 ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        Conv2d = get_same_padding_conv2d()
        self.activation = MemoryEfficientSwish()
        norm = nn.InstanceNorm2d

        self.input_conv = Conv2d(in_channels=n_channels,
                                 out_channels=n,
                                 kernel_size=3,
                                 stride=1,
                                 bias=bias,
                                 )

        self.inc = BasicBlock(n, norm_layer=norm, activation_func=self.activation, bias=bias)

        self.down1_norm = norm(n)
        self.down1 = Conv2d(in_channels=n,
                            out_channels=2*n,
                            kernel_size=3,
                            stride=2,
                            bias=bias,
                            )

        self.down1_basic = BasicBlock(2*n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.down1_bottle = nn.Sequential(*[Bottleneck(channels=2*n,
                                  mid_channels=2*n // 2,
                                  ratio=ratio,
                                  norm_layer=norm,
                                  activation_func=self.activation,
                                  bias=bias,
                                  ) for i in range(n_repeats)])

        self.down2_norm = norm(2*n)
        self.down2 = Conv2d(in_channels=2*n,
                            out_channels=4*n,
                            kernel_size=3,
                            stride=2,
                            bias=bias,
                            )
        self.down2_basic = BasicBlock(4*n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.down2_bottle = nn.Sequential(*[Bottleneck(channels=4*n,
                                  mid_channels=4*n // 4,
                                  ratio=ratio,
                                  norm_layer=norm,
                                  activation_func=self.activation,
                                  bias=bias,
                                  ) for i in range(n_repeats)])

        self.down3_norm = norm(4*n)
        self.down3 = Conv2d(in_channels=4*n,
                            out_channels=8*n,
                            kernel_size=3,
                            stride=2,
                            bias=bias,
                            )
        self.down3_basic = BasicBlock(8*n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.down3_bottle = nn.Sequential(*[Bottleneck(channels=8*n,
                                  mid_channels=8*n // 4,
                                  ratio=ratio,
                                  norm_layer=norm,
                                  activation_func=self.activation,
                                  bias=bias,
                                  ) for i in range(n_repeats)])

        self.down4_norm = norm(8*n)
        self.down4 = Conv2d(in_channels=8*n,
                            out_channels=8*n,
                            kernel_size=3,
                            stride=2,
                            bias=bias,
                            )

        self.down4_basic = BasicBlock(8*n, norm_layer=norm, activation_func=self.activation, bias=bias)

        # Layer furthest down
        self.bottle_middle_1 = nn.Sequential(*[Bottleneck(channels=8*n,
                                        mid_channels=8*n // 4,
                                        ratio=ratio,
                                        norm_layer=norm,
                                        activation_func=self.activation,
                                        bias=bias,
                                        ) for i in range(n_repeats // 2)])

        # BiFPN layer as suggested by EfficientDet by tensorflow
        self.bifpns = nn.ModuleList([
            BiFPN(channels=[n, 2*n, 4*n, 8*n, 8*n], layers=5)
            for i in range(BiFPN_layers)])


        self.bottle_middle_2 = nn.Sequential(*[Bottleneck(channels=8*n,
                                        mid_channels=8*n // 4,
                                        ratio=ratio,
                                        norm_layer=norm,
                                        activation_func=self.activation,
                                        bias=bias,
                                        ) for i in range(n_repeats // 2)])

        # Everything from here on must be fixed... woop woop

        self.up4_basic = BasicBlock(8*n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.up4_norm = norm(8*n)
        self.up4 = nn.ConvTranspose2d(in_channels=8*n,
                                        out_channels=8*n,
                                        kernel_size=2,
                                        stride=2,
                                        bias=bias,
                                        )


        self.up3_channel_norm = norm(2*8*n)
        self.up3_channel = Conv2d(in_channels=2*8*n,
                                  out_channels=8*n,
                                  kernel_size=3,
                                  stride=1,
                                  bias=bias,
                                  )


        self.up3_bottle = nn.Sequential(*[Bottleneck(channels=8*n,
                                     mid_channels=8*n // 4,
                                     ratio=ratio,
                                     norm_layer=norm,
                                     activation_func=self.activation,
                                     bias=bias,
                                     ) for i in range(n_repeats)])
        self.up3_basic = BasicBlock(8*n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.up3_norm = norm(8*n)
        self.up3 = nn.ConvTranspose2d(in_channels=8*n,
                                      out_channels=4*n,
                                      kernel_size=2,
                                      stride=2,
                                      bias=bias,
                                      )


        self.up2_channel_norm = norm(2*4*n)
        self.up2_channel = Conv2d(in_channels=2*4*n,
                                  out_channels=4*n,
                                  kernel_size=3,
                                  stride=1,
                                  bias=bias,
                                  )


        self.up2_bottle = nn.Sequential(*[Bottleneck(channels=4*n,
                                     mid_channels=4*n // 4,
                                     ratio=ratio,
                                     norm_layer=norm,
                                     activation_func=self.activation,
                                     bias=bias,
                                     ) for i in range(n_repeats)])

        self.up2_basic = BasicBlock(4*n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.up2_norm = norm(4*n)
        self.up2 = nn.ConvTranspose2d(in_channels=4*n,
                                      out_channels=2*n,
                                      kernel_size=2,
                                      stride=2,
                                      bias=bias,
                                      )


        self.up1_channel_norm = norm(2*2*n)
        self.up1_channel = Conv2d(in_channels=2*2*n,
                                  out_channels=2*n,
                                  kernel_size=3,
                                  stride=1,
                                  bias=bias,
                                  )


        self.up1_bottle = nn.Sequential(*[Bottleneck(channels=2*n,
                                     mid_channels=2*n // 2,
                                     ratio=ratio,
                                     norm_layer=norm,
                                     activation_func=self.activation,
                                     bias=bias,
                                     ) for i in range(n_repeats)])
        self.up1_basic = BasicBlock(2*n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.up1_norm = norm(2*n)
        self.up1 = nn.ConvTranspose2d(in_channels=2*n,
                                      out_channels=n,
                                      kernel_size=2,
                                      stride=2,
                                      bias=bias,
                                      )

        self.out_channel_norm = norm(2*n)
        self.out_channel = Conv2d(in_channels=2*n,
                                  out_channels=n,
                                  kernel_size=3,
                                  stride=1,
                                  bias=bias,
                                  )
        self.out_1 = BasicBlock(n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.final_bottle = Bottleneck(channels=n,
                                       mid_channels=n,
                                       ratio=ratio,
                                       norm_layer=norm,
                                       activation_func=self.activation,
                                       bias=bias,
                                       )
        self.out_2 = BasicBlock(n, norm_layer=norm, activation_func=self.activation, bias=bias)
        self.outc = nn.Conv2d(in_channels=n, out_channels=1, stride=1, kernel_size=1)

    def forward(self, x):

        x = self.input_conv(x)
        x1 = self.inc(x)


        x2 = self.down1(self.activation(self.down1_norm(x1)))
        x2 = self.down1_basic(x2)
        x2 = self.down1_bottle(x2)

        x3 = self.down2(self.activation(self.down2_norm(x2)))
        x3 = self.down2_basic(x3)
        x3 = self.down2_bottle(x3)

        x4 = self.down3(self.activation(self.down3_norm(x3)))
        x4 = self.down3_basic(x4)
        x4 = self.down3_bottle(x4)

        x = self.down4(self.activation(self.down4_norm(x4)))
        x = self.down4_basic(x)
        x = self.bottle_middle_1(x)

        inputs = [x1, x2, x3, x4, x]
        for bifpn in self.bifpns:
            inputs = bifpn(inputs)
            inputs[0] += x1
            inputs[1] += x2
            inputs[2] += x3
            inputs[3] += x4
            inputs[4] += x
            x1, x2, x3, x4, x = inputs

        x = self.bottle_middle_2(x)
        x = self.up4_basic(x)
        x = self.up4(self.activation(self.up4_norm(x)))

        x = torch.cat([x, x4], dim=1)
        x = self.up3_channel(self.activation(self.up3_channel_norm(x)))
        x = self.up3_bottle(x)
        x = self.up3_basic(x)
        x = self.up3(self.activation(self.up3_norm(x)))

        x = torch.cat([x, x3], dim=1)
        x = self.up2_channel(self.activation(self.up2_channel_norm(x)))
        x = self.up2_bottle(x)
        x = self.up2_basic(x)
        x = self.up2(self.activation(self.up2_norm(x)))

        x = torch.cat([x, x2], dim=1)
        x = self.up1_channel(self.activation(self.up1_channel_norm(x)))
        x = self.up1_bottle(x)
        x = self.up1_basic(x)
        x = self.up1(self.activation(self.up1_norm(x)))

        x = torch.cat([x, x1], dim=1)
        x = self.out_channel(self.activation(self.out_channel_norm(x)))
        x = self.out_1(x)
        x = self.final_bottle(x)
        x = self.out_2(x)

        return self.outc(x)
