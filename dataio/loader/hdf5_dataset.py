import torch.utils.data as data
import numpy as np
import datetime

import h5py
from os.path import join


class HDF5Dataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(HDF5Dataset, self).__init__()
        if split == "validation" or split == "val":
            split = "val"
        dataPath = join(root_dir, split)
        dataPath = dataPath + ".hdf5"
        print(split+" dataPath: ", dataPath)
        # open HDF5 files for reading
        self.data = h5py.File(dataPath)

        # data augmentation
        self.transform = transform


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        #input, target = self.data["data"][index], self.data["seg"][index]
        input, target = np.squeeze(self.data["data"][index]), np.squeeze(self.data["seg"][index])

        # do transformations
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return self.data["data"].shape[0]
