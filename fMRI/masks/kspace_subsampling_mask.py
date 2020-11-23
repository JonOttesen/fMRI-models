import contextlib

import torch
import numpy as np


@contextlib.contextmanager
def temp_seed(seed):
    """
    Source:
    https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class KspaceMask:
    """
    Creates a sub-sampling mask for the k_space data
    There are three different types of sub-sampling masks:
    mask_random_uniform: returns a mask where all the data points except
        the center are uniformly randomly sampled
    """
    MASK_TYPES = ['equidistant', 'random']

    def __init__(self,
                 acceleration: int,
                 mask_type: str = 'equidistant',
                 seed: int = None,
                 center_fraction: float = 0.08,
                 ):

        self.acceleration = acceleration
        self._mask_type = mask_type
        self.seed = seed
        self.center_fraction = center_fraction

        assert mask_type in self.MASK_TYPES, 'mask_type {} not in MASK_TYPES {}'.format(mask_type, self.MASK_TYPES)

        self.mask_type_func = {
            'equidistant': self.equidistant,
            'random': self.random_uniform,
            }
        self.masks = None if seed == None else {}  # Dict for storing prior created masks

    @property
    def mask_type(self):
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value: str):
        self._mask_type = value
        # Using setter to ensure correct string type
        assert value in self.MASK_TYPES, 'mask_type {} not in MASK_TYPES {}'.format(value, self.MASK_TYPES)

    def __call__(self, lines: int):
        """
        Wrapper for the mask generator calling either:
        _mask_random_uniform or _mask_linearly_spaced depending on the mask type
        Args:
            lines: (int), the number of columns the mask is used for i.e k_x lines
        returns: (torch.Tensor), shape: (lines)
        """
        # No point recreating the same masks every time if I am using a seed anyway
        if self.masks is not None:
            if lines not in self.masks.keys():
                self.masks[lines] = self.mask_type_func[self.mask_type](lines)
            return self.masks[lines]
        return self.mask_type_func[self.mask_type](lines)

    def random_uniform(self, lines: int) -> torch.Tensor:
        """
        Creates a mask by selecting uniformly random which columns that is included in the mask,
        except for the low frequency center.
        There are a total of lines/self.acceleration masks
        Args:
            lines: (int), the number of columns the mask is used for i.e k_x lines
        returns: (torch.Tensor), shape: (lines)
        """

        with temp_seed(self.seed):
            mask = np.zeros(lines)
            indices = np.arange(lines)

            center_frac = self.center_fraction

            low_freq = int(round(center_frac/2*lines))

            k_0 = int(lines/2)
            high_freq = int(lines/self.acceleration) - low_freq*2

            mask[k_0 - low_freq:k_0 + low_freq] = 1
            indices = indices[mask != 1]

            indices = np.random.choice(a=indices, size=high_freq, replace=False)
            mask[indices] = 1

        return torch.from_numpy(mask).bool()

    def equidistant(self, lines: int) -> torch.Tensor:
        """
        Creates a mask by with linearly spaced columns except the low frequency center
        There should be a total of lines/self.acceleration masks (pm 1-2)
        Args:
            lines: (int), the number of columns the mask is used for i.e k_x lines
        returns:
            returns: (torch.Tensor), shape: (lines)
        """

        with temp_seed(self.seed):
            mask = np.zeros(lines)

            center_frac = self.center_fraction

            low_freq = int(round(center_frac/2*lines))  # Low freq lines on each side of origin

            k_0 = int(lines/2)
            high_freq = int(lines/self.acceleration) - low_freq*2

            mask[k_0 - low_freq:k_0 + low_freq] = 1

            step = (k_0 - low_freq)/(high_freq/2)

            # Calculating the indices on the left and right side of k-space origin
            left_low_freq = (np.random.uniform(0, self.acceleration - 1)
                            + np.arange(int(high_freq/2))*step).astype(dtype=np.int16)
            right_low_freq = (k_0 + low_freq + np.random.uniform(1, self.acceleration)\
                             + np.arange(int(high_freq/2))*step).astype(dtype=np.int16)

            # Fills the mask with the linear filter
            mask[left_low_freq] = 1
            mask[right_low_freq] = 1

        return torch.from_numpy(mask).bool()
