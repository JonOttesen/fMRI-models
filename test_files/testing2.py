from copy import deepcopy

import h5py
import numpy as np
from matplotlib import pyplot as plt

import fastmri
from fastmri.data import transforms as T

import warnings
warnings.filterwarnings("ignore")

# from dataset import *

from fMRI_dataset_processing import test_main


exit()

def slice_to_image(k_slice):
    a = fastmri.ifft2c(T.to_tensor(k_slice))
    b = fastmri.complex_abs(a)
    c = fastmri.rss(b, dim=0)
    return c


def freq_correction(k_slice):
    k_slice = deepcopy(k_slice)
    slice_conjugate = np.conj(k_slice)

    k_x = k_slice.shape[2]
    k_y = k_slice.shape[1]
    kx_0 = int(k_x/2)
    ky_0 = int(k_y/2)

    low_freq = 13

    complex_conj = np.flip(np.conj(deepcopy(k_slice)), axis=(1, 2))

    complex_conj[:, :, :kx_0-13] = 0
    complex_conj[:, :, kx_0:] = 0

    k_slice[:, :, :kx_0-13] = 0
    k_slice[:, :, kx_0:] = 0

    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(slice_to_image(complex_conj).numpy()), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(slice_to_image(k_slice).numpy()), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(slice_to_image(k_slice).numpy() - slice_to_image(complex_conj).numpy()), cmap='gray')
    plt.show()

    diff = fastmri.ifft2c(T.to_tensor(k_slice)) - fastmri.ifft2c(T.to_tensor(complex_conj))
    diff = fastmri.tensor_to_complex_np(fastmri.fft2c(diff))
    diff2 = k_slice - complex_conj

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(slice_to_image(diff).numpy()), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(slice_to_image(diff2).numpy()), cmap='gray')

    plt.show()
    print(k_slice.shape)

    for i in range(3):
        sp = np.fft.fft(k_slice[i*3])
        # freq = np.fft.fftfreq(n=k_x, d=1)
        freq = np.arange(k_x)
        plt.plot(freq, np.sum(sp.real, axis=0))
    plt.show()
    plt.plot(freq, np.sum(sp.imag, axis=0))
    plt.show()
    return complex_conj



def complex_conjugate(k_slice, mask):
    k_slice = deepcopy(k_slice)
    slice_conjugate = np.conj(k_slice)

    k_x = k_slice.shape[2]
    k_y = k_slice.shape[1]
    kx_0 = int(k_x/2)
    ky_0 = int(k_y/2)

    low_freq = 13

    if mask == None:
        mask = np.zeros(k_x)
    mask_t = mask != 1

    complex_conj = np.zeros_like(k_slice)

    # Flipping each quadrant individually
    complex_conj[:, ky_0:, :kx_0] = np.flip(deepcopy(slice_conjugate[:, :ky_0, kx_0:]), axis=(1, 2))
    complex_conj[:, ky_0:, kx_0:] = np.flip(deepcopy(slice_conjugate[:, :ky_0, :kx_0]), axis=(1, 2))
    complex_conj[:, :ky_0, :kx_0] = np.flip(deepcopy(slice_conjugate[:, ky_0:, kx_0:]), axis=(1, 2))
    complex_conj[:, :ky_0, kx_0:] = np.flip(deepcopy(slice_conjugate[:, ky_0:, :kx_0]), axis=(1, 2))

    k_slice[:, :, mask_t] = complex_conj[:, :, mask_t]

    """
    quad_1 = np.flip(deepcopy(slice_conjugate[:, :ky_0, kx_0+low_freq:]), axis=(1, 2))
    quad_2 = np.flip(deepcopy(slice_conjugate[:, :ky_0, :kx_0-low_freq]), axis=(1, 2))
    quad_3 = np.flip(deepcopy(slice_conjugate[:, ky_0:, kx_0+low_freq:]), axis=(1, 2))
    quad_4 = np.flip(deepcopy(slice_conjugate[:, ky_0:, :kx_0-low_freq]), axis=(1, 2))

    k_slice[:, ky_0:, :kx_0-low_freq][:, :, mask_t[:kx_0-low_freq]] = quad_1[:, :, mask_t[:kx_0-low_freq]]
    k_slice[:, ky_0:, kx_0+low_freq:][:, :, mask_t[kx_0+low_freq:]] = quad_2[:, :, mask_t[kx_0+low_freq:]]
    k_slice[:, :ky_0, :kx_0-low_freq][:, :, mask_t[:kx_0-low_freq]] = quad_3[:, :, mask_t[:kx_0-low_freq]]
    k_slice[:, :ky_0, kx_0+low_freq:][:, :, mask_t[kx_0+low_freq:]] = quad_4[:, :, mask_t[kx_0+low_freq:]]
    """
    return k_slice


a = DatasetContainer()

path = '/home/jon/Documents/CRAI/fMRI/testing/file_brain_AXT1_202_2020034.h5'

a.add_entry(entry=DatasetEntry(image_path=path,
                               datasetname='fMRI',
                               dataset_type='training',
                               sequence_type='FLAIR',
                               field_strength=3,
                               pre_contrast=True,
                               multicoil=True))

a.add_info(info=DatasetInfo(datasetname='fMRI',
                            dataset_type='training',
                            source='fMRI challenge',
                            dataset_description='Data given in the fMRI challenge'))

hf = h5py.File(a[0].image_path)
volume = hf['kspace'][:]


slice_kspace = volume[10]

slice_kspace_2 = freq_correction(slice_kspace)

# slice_kspace_2 = complex_conjugate(k_slice=deepcopy(slice_kspace), mask=None)
# slice_kspace_2 = slice_kspace

def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
    plt.show()

show_coils(np.log(np.abs(slice_kspace_2) + 1e-9), [0, 5, 10])  # This shows coils 0, 5 and 10



slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image

# show_coils(slice_image_abs, [0, 5, 10], cmap='gray')

slice_image_rss = fastmri.rss(slice_image_abs, dim=0)

slice_kspace22 = T.to_tensor(slice_kspace_2)      # Convert from numpy array to pytorch tensor
slice_image2 = fastmri.ifft2c(slice_kspace22)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs2 = fastmri.complex_abs(slice_image2)   # Compute absolute value to get a real image

slice_image_rss2 = fastmri.rss(slice_image_abs2, dim=0)

slice_kspace[:, :, int(slice_kspace.shape[2]/2)+13:] = 0
slice_kspace[:, :, :int(slice_kspace.shape[2]/2) - 13] = 0
a = fastmri.ifft2c(T.to_tensor(slice_kspace))
b = fastmri.complex_abs(a)
c = fastmri.rss(b, dim=0)

fig = plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(slice_image_rss2.numpy()), cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(c.numpy()), cmap='gray')
plt.show()
