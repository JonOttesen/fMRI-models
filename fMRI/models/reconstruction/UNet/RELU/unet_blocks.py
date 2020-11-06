import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=mid_channels,
                               kernel_size=3,
                               padding=1)

        self.norm1 = nn.InstanceNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(in_channels=mid_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1)

        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.ReLU(inplace=True)(x)

        return x



class DownConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):

        x1 = self.up(x1)

        dY = x2.size()[2] - x1.size()[2]
        dX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [dX // 2, dX - dX // 2,
                        dY // 2, dY - dY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
