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
    se_ratio: Union[float, None]
    id_skip: bool

# Efficientnet blocks
blocks = [
    BlockArgs(1, 3, [1, 1], 1, 32, 16, 0.25, True),
    BlockArgs(2, 3, [2, 2], 6, 16, 24, 0.25, True),
    BlockArgs(2, 5, [2, 2], 6, 24, 40, 0.25, True),
    BlockArgs(3, 3, [2, 2], 6, 40, 80, 0.25, True),
    BlockArgs(3, 5, [1, 1], 6, 80, 112, 0.25, True),
    BlockArgs(4, 5, [2, 2], 6, 112, 192, 0.25, True),
    BlockArgs(1, 3, [1, 1], 6, 192, 320, 0.25, True),
    ]

@dataclass
class GlobalParams:
    width_coefficient: int
    depth_coefficient: int
    image_size: int
    min_depth: int
    dropout_rate: float = 0.2
    batch_norm_momentum: float = 0.99
    batch_norm_epsilon: float = 0.001
    drop_connect_rate: float = 0.2
    depth_division: int = 8
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
