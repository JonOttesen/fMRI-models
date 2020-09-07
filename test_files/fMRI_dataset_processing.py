import h5py
import sys
import blosc
import time
import io

import tarfile

def test_main(name):
    print(name)
    print(__name__)


def dataset_reader_writer(dataset, output):

    tar = tarfile.open(dataset)

    for tarinfo in tar:

        print(tarinfo.name, "is", tarinfo.size, "bytes in size and is ", end="")

        if tarinfo.isreg():
            print("a regular file.")
            tar_file = tar.extractfile(member=tarinfo)
            tar_bytes = io.BytesIO()

            for byte_sequence in tar_file:
                tar_bytes.write(byte_sequence)

            hf = h5py.File(tar_bytes)

        elif tarinfo.isdir():
            print("a directory.")
        else:
            print("something else.")

    tar.close()
