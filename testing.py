import time
import numpy as np

from fMRI.masks import KspaceMask


img = np.random.randint(0, 255, (320, 320))

mask_generator = KspaceMask(acceleration=4, seed=42)
mask = mask_generator._mask_linearly_spaced(lines=320)

N = int(1e6)

start_time = time.time()

# for i in range(N):
    # img[mask]

print('Time wo generating masks for {} subsamplings: {}'.format(N, time.time() - start_time))


start_time = time.time()

for i in range(N):
    mask = mask_generator.mask(lines=320)
    img[mask]

print('Time with generating masks for {} subsamplings: {}'.format(N, time.time() - start_time))

