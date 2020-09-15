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

    def __init__(self,
                 acceleration: int,
                 center_fraction: float = 0.08,
                 randomize_center_fraction: bool = False,
                 randomize_center_interval: float = 0.05):

        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.randomize_center_fraction = randomize_center_fraction
        self.randomize_center_interval = randomize_center_interval

        if self.randomize_center_interval > self.center_fraction:
            assert ValueError('randomize_center_interval is larger than center_interval')


    def mask_random_uniform(self, lines: int, seed: int = None) -> torch.Tensor:
        """
        Creates a mask by selecting uniformly random which columns that is included in the mask,
        except for the low frequency center.
        There are a total of lines/self.acceleration masks
        Args:
            lines: (int), the number of columns the mask is used for i.e k_x lines
            seed: (int), integer number for the seed, the cool kids use 42
        returns: (torch.Tensor), shape: (lines)
        """

        with temp_seed(seed):
            mask = np.zeros(lines)
            indices = np.arange(lines)

            if self.randomize_center_fraction:
                center_frac = np.random.uniform(self.center_fraction - self.randomize_center_interval,
                                                self.center_fraction + self.randomize_center_interval)
                low_freq = int(round(center_frac/2*lines))
            else:
                low_freq = int(round(self.center_fraction/2*lines))

            k_0 = int(lines/2)
            high_freq = int(lines/self.acceleration) - low_freq*2

            mask[k_0 - low_freq:k_0 + low_freq] = 1
            indices = indices[mask != 1]

            indices = np.random.choice(a=indices, size=high_freq, replace=False)
            mask[indices] = 1

        return torch.from_numpy(mask).bool()


    def mask_linearly_spaced(self, lines: int, seed: int = None) -> torch.Tensor:
        """
        Creates a mask by with linearly spaced columns except the low frequency center
        There should be a total of lines/self.acceleration masks (pm 1-2)
        Args:
            lines: (int), the number of columns the mask is used for i.e k_x lines
            seed: (int), integer number for the seed, the cool kids use 42
        returns:
            returns: (torch.Tensor), shape: (lines)
        """

        with temp_seed(seed):
            mask = np.zeros(lines)

            if self.randomize_center_fraction:
                center_frac = np.random.uniform(self.center_fraction - self.randomize_center_interval,
                                                self.center_fraction + self.randomize_center_interval)
                low_freq = int(round(center_frac/2*lines))  # Low freq lines on each side of origin
            else:
                low_freq = int(round(self.center_fraction/2*lines))  # Low freq lines on each side of origin

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
