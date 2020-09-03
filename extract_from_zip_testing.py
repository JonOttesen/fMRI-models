import h5py
import sys
import blosc
import time
import io
import numpy as np
import base64

from copy import deepcopy

import tarfile
import warnings
warnings.filterwarnings("ignore")

compression_orig = 'file_brain_AXFLAIR_200_6002441.h5'
compression_test = 'testing_compressed.h5'
dataset = 'brain_multicoil_test.tar.gz'

#If any compression is to be used, the image has to be compressed beforehand and the compressed is saved


# hf = h5py.File(compression_test)
# print(list(hf.keys()))
# print(hf['kspace'][0])
# print(hf['mask'].shape)
"""

start = time.time()
hf = h5py.File(compression_orig)
for i in range(500):
    k_space = hf['kspace'][0]
print('Time wo compression: ', time.time() - start)

start = time.time()
hf = h5py.File(compression_test)
for i in range(500):
    k_space = blosc.decompress(base64.b64decode(hf['kspace'][0].encode('utf-8')))
print('Time with compression: ', time.time() - start)

exit()
"""

# """
patient_id = list()
tar = tarfile.open(dataset)

for tarinfo in tar:

    if tarinfo.isreg():

        hello_world = tar.extractfile(member=tarinfo)

        f = io.BytesIO()
        for hello in hello_world:
            f.write(hello)
        hf = h5py.File(f, 'r')
        # print(hf.keys())
        info = dict(hf.attrs)
        if info['patient_id'] not in patient_id:
            patient_id.append(info['patient_id'])
        else:
            print('This is verrrrrryyyyyyy noice')
        print(dict(hf.attrs))

        # print(hf['ismrmrd_header'][()].decode('utf-8'))






        """
        k_space = deepcopy(hf['kspace'][:])
        n = len(k_space)
        new_hf = h5py.File(compression_test)
        dt = h5py.string_dtype(encoding='utf-8')
        dset = new_hf.create_dataset('kspace', shape=(n,), dtype=dt)
        for i in range(n):
            k_bytes = k_space[i].tobytes()
            k_comp = blosc.compress(k_bytes, typesize=8, clevel=9, cname='blosclz')
            # dset[i] = base64.b64encode(k_comp).decode('utf-8')
            dset[i] = np.void(k_comp)
        # dset.attrs['test'] = base64.b64encode(k_comp).decode('utf-8')
        # new_hf.create_dataset('kspace', data=np.void(compressed_bytes))
        hf.close()
        tar.close()
        new_hf.close()
        exit()
        """

# """

"""

hf = h5py.File(file_name)

print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))

k_space = hf['kspace'][:]
print(k_space.shape)

print('MB:', sys.getsizeof(k_space)/1024**2)
print('Compressed:', sys.getsizeof(blosc.compress(k_space, typesize=1, cname='blosclz'))/1024**2)

hf = h5py.File(file_name2)

print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))

k_space = hf['kspace'][:]
print(k_space.shape)

print('MB:', sys.getsizeof(k_space)/1024**2)
print('Compressed:', sys.getsizeof(blosc.compress(k_space, typesize=1, cname='blosclz'))/1024**2)

k_space.tobytes()
comp = np.void(blosc.compress(k_space, typesize=1, cname='blosclz'))
a = blosc.decompress(comp)

plt.imshow(a)
plt.show()

bio = io.BytesIO()
with h5py.File(bio) as f:
    f['data'] = comp

data = bio.getvalue() # data is a regular Python bytes object.
print("Total size:", len(data)/1024**2)

# file = 'multicoil_test/file_brain_AXFLAIR_200_6002441.h5'

# a = gzip.open(dataset_name, 'r')
# for i, line in enumerate(a):
#     pass
# print(i)

"""
