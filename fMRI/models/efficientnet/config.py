from dataclasses import dataclass
from typing import Union, List, Tuple


@dataclass
class BlockArgs:
    num_repeat: int
    kernel_size: int
    stride: Union[int, Tuple[int, int]]
    expand_ratio: int
    input_filters: int
    output_filters: int
    se_ratio: Union[float, None] = 0.25
    id_skip: bool = True

# Efficientnet blocks
blocks = [
    BlockArgs(
        num_repeat=1,
        kernel_size=3,
        stride=[1, 1],
        expand_ratio=1,
        input_filters=32,
        output_filters=16,
    ),
    BlockArgs(
        num_repeat=2,
        kernel_size=3,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=16,
        output_filters=24,
    ),
    BlockArgs(
        num_repeat=2,
        kernel_size=5,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=24,
        output_filters=40,
    ),
    BlockArgs(
        num_repeat=3,
        kernel_size=3,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=40,
        output_filters=80,
    ),
    BlockArgs(
        num_repeat=3,
        kernel_size=5,
        stride=[1, 1],
        expand_ratio=6,
        input_filters=80,
        output_filters=112,
    ),
    BlockArgs(
        num_repeat=4,
        kernel_size=5,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=112,
        output_filters=192,
    ),
    BlockArgs(
        num_repeat=1,
        kernel_size=3,
        stride=[1, 1],
        expand_ratio=6,
        input_filters=192,
        output_filters=320,
    ),
    ]

@dataclass
class GlobalParams:
    width_coefficient: Union[int, None] = None
    depth_coefficient: Union[int, None] = None
    image_size: Union[int, None] = None
    min_depth: Union[int, None] = None
    dropout_rate: float = 0.2
    batch_norm_momentum: float = 0.99
    batch_norm_epsilon: float = 0.001
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    include_top: bool = True


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.
    Args:
        model_name (str): Model name to be queried.
    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
    params_dict = {
        'efficientnet-b0':
            {'width': 1.0, 'depth': 1.0, 'resolution': 224, 'dropout': 0.2},
        'efficientnet-b1':
            {'width': 1.0, 'depth': 1.1, 'resolution': 240, 'dropout': 0.2},
        'efficientnet-b2':
            {'width': 1.1, 'depth': 1.2, 'resolution': 260, 'dropout': 0.3},
        'efficientnet-b3':
            {'width': 1.2, 'depth': 1.4, 'resolution': 300, 'dropout': 0.3},
        'efficientnet-b4':
            {'width': 1.4, 'depth': 1.8, 'resolution': 380, 'dropout': 0.4},
        'efficientnet-b5':
            {'width': 1.6, 'depth': 2.2, 'resolution': 456, 'dropout': 0.4},
        'efficientnet-b6':
            {'width': 1.8, 'depth': 2.6, 'resolution': 528, 'dropout': 0.5},
        'efficientnet-b7':
            {'width': 2.0, 'depth': 3.1, 'resolution': 600, 'dropout': 0.5},
        'efficientnet-b8':
            {'width': 2.2, 'depth': 3.6, 'resolution': 672, 'dropout': 0.5},
        'efficientnet-l2':
            {'width': 4.3, 'depth': 5.3, 'resolution': 800, 'dropout': 0.5},
        }

    return params_dict[model_name]
