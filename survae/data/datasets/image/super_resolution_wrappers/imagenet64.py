import os
import torch
import torch.utils.data as data
import numpy as np
import errno
import tarfile
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import resize

from survae.data import DATA_PATH
from survae.data.datasets.image.unsupervised_wrappers.imagenet64 import UnsupervisedImageNet64Dataset


class SuperResolutionImageNet64Dataset(UnsupervisedImageNet64Dataset):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False, sr_scale_factor=4, bicubic=False):

        if transform is None:
            transform = Compose([ToTensor()])
        
        super(SuperResolutionImageNet64Dataset, self).__init__(root,
                                                               train=train,
                                                               transform=transform,
                                                               download=download)
        
        assert isinstance(sr_scale_factor, int) and sr_scale_factor > 1
        self.sr_scale_factor = sr_scale_factor
        self.bicubic = bicubic

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        hr = super(SuperResolutionImageNet64Dataset, self).__getitem__(index)

        if self.bicubic:
            lr = resize(hr, hr.shape[0] // self.sr_scale_factor, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            # h_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # w_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # lr = hr[:, h_offset::self.sr_scale_factor, w_offset::self.sr_scale_factor]
            lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
            
        return (hr, lr)
