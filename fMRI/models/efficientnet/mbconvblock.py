"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .swish import (
    Swish,
    MemoryEfficientSwish,
    )
from .utils import (
    drop_connect,
    get_same_padding_conv2d,
    calculate_output_image_size
)

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self,
                 block_args,
                 global_params,
                 image_size=None,
                 ):
        super().__init__()

        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self.batch_norm_epsilon = global_params.batch_norm_epsilon


        self.has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
        self.expand_ratio = block_args.expand_ratio

        # Expansion phase (Inverted Bottleneck)
        in_channels = block_args.input_filters  # number of input channels
        out_channels = block_args.input_filters * self.expand_ratio  # number of output channels


        if self.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self.expand_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(num_features=out_channels,
                                      momentum=self.batch_norm_momentum,
                                      eps=self.batch_norm_epsilon,
                                      )

        # Depthwise convolution phase
        kernel_size = block_args.kernel_size
        stride = block_args.stride

        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.depthwise_conv = Conv2d(
            in_channels=out_channels, out_channels=out_channels, groups=out_channels,  # groups makes it depthwise
            kernel_size=kernel_size, stride=stride, bias=False)

        # ------------------------------

        self.bn1 = nn.BatchNorm2d(num_features=out_channels,
                                  momentum=self.batch_norm_momentum,
                                  eps=self.batch_norm_epsilon,
                                  )
        image_size = calculate_output_image_size(image_size, stride)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(block_args.input_filters * block_args.se_ratio))
            self.se_reduce = Conv2d(in_channels=out_channels, out_channels=num_squeezed_channels, kernel_size=1)
            self.se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=out_channels, kernel_size=1)

        # Pointwise convolution phase
        final_output = block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.project_conv = Conv2d(in_channels=out_channels,
                                   out_channels=final_output,
                                   kernel_size=1,
                                   bias=False,
                                   )
        self.bn2 = nn.BatchNorm2d(num_features=final_output,
                                  momentum=self.batch_norm_momentum,
                                  eps=self.batch_norm_epsilon,
                                  )
        self.swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self.expand_conv(inputs)
            x = self.bn0(x)
            x = self.swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self.se_reduce(x_squeezed)
            x_squeezed = self.swish(x_squeezed)
            x_squeezed = self.se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self.project_conv(x)
        x = self.bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()
