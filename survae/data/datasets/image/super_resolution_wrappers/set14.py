import os
import torch
import torchvision
import torch.utils.data as data
import numpy as np
import pandas as pd
import errno
import zipfile
import io
from PIL import Image
from torchvision.transforms.functional import resize
from survae.data import DATA_PATH


class Set14Dataset(data.Dataset):
    """
    http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip
    """
    def __init__(self, resize_hw=None, root=DATA_PATH, split='test', transform=None, sr_scale_factor=4, bicubic=True, repeats=1):
        super(Set14Dataset, self).__init__()
        
        assert split in {'test'}
        self.root = os.path.expanduser(root)
        self.raw_data_folder = os.path.join(self.root, 'Set14/')
        self.split = split
        self.transform = transform
        self.sr_scale_factor = sr_scale_factor
        self.resize_hw = resize_hw
        self.bicubic = bicubic
        self.files = [os.path.join(self.raw_data_folder, file) for file in os.listdir(self.raw_data_folder)] * repeats

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """
        hr = Image.open(self.files[index]).convert('RGB')

        if self.resize_hw:
            hr = resize(hr, size=(self.resize_hw, self.resize_hw), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        if self.transform is not None:
            hr = self.transform(hr)

        if self.bicubic:
            lr = resize(hr, hr.shape[1] // self.sr_scale_factor, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        
        return (hr, lr)

    def __len__(self):
        return len(self.files)


