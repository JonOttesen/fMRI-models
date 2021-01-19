import torch
import torch.nn.functional as F
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

from fMRI.models.blocks.lstm_cell import ConvLSTMCell


class ULSTMNet(nn.Module):

    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 n_lsmt: int = 20,
                 n: int = 128,
                 n_repeats: int = 1,
                 BiFPN_layers: int = 0,
                 ratio: float = 1./8,
                 bias: bool = False,
                 ):
        super().__init__()

        self.encoder_image = self.Encoder(
            n_channels=n_channels,
            n=n,
            n_repeats=n_repeats,
            BiFPN_layers=BiFPN_layers,
            ratio=ratio,
            bias=bias,
            )
        self.encoder_coil = self.Encoder(
            n_channels=2,
            n=n,
            n_repeats=n_repeats,
            BiFPN_layers=BiFPN_layers,
            ratio=ratio,
            bias=bias,
            )

        self.lstm = ConvLSTMCell(
            input_dim=n*8,
            hidden_dim=n*8,
            kernel_size=3,
            bias=True,
            )

        self.decoder = Decoder(
            n_classes=n_classes,
            n=n,
            n_repeats=n_repeats,
            ratio=ratio,
            bias=bias,
            )

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
