from typing import Optional, Callable, Union, Tuple

import torch
import torch.nn as nn

from ..blocks.utils import get_same_padding_conv2d, get_same_padding_maxPool2d


class Bottleneck(nn.Module):
    """
    Original paper:
    https://arxiv.org/pdf/1603.05027.pdf
    Inspiration from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_func: Optional[Callable[..., nn.Module]] = None,
                 ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        stride = stride if isinstance(stride, (list, tuple)) else (stride, stride)
        if stride[0] > 1 or stride[1] > 1 and downsample is None:
            downsample = get_same_padding_maxPool2d()
            downsample = downsample(kernel_size=stride, stride=stride)

        Conv2d = get_same_padding_conv2d(image_size=None)
        self.norm0 = norm_layer(in_channels)

        self.conv1 = Conv2d(in_channels, mid_channels, kernel_size=1)
        self.norm1 = norm_layer(mid_channels)

        self.conv2 = Conv2d(in_channels=mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            stride=stride,
                            groups=groups,
                            dilation=dilation,
                            )
        self.norm2 = norm_layer(mid_channels)

        self.conv3 = Conv2d(mid_channels, out_channels, kernel_size=1)

        if activation_func is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation_func

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm0(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)

        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity

        return x
