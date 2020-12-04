from typing import Optional, Callable

import torch
import torch.nn as nn

from .conv_pad import get_same_padding_conv2d


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ) -> None:

        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        Conv2d = get_same_padding_conv2d(image_size=None)
        self.conv1 = Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=stride,
                            bias=False,
                            )
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=stride,
                            bias=False,
                            )
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
