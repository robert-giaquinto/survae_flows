import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import errno
import zipfile
import io
from PIL import Image
from torchvision.transforms.functional import crop, resize

from survae.data import DATA_PATH
from survae.data.datasets.image.unsupervised_wrappers.celeba import UnsupervisedCelebA32Dataset, UnsupervisedCelebA64Dataset, UnsupervisedCelebA128Dataset


class SuperResolutionCelebA32Dataset(UnsupervisedCelebA32Dataset):
    """
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    """
    raw_folder = 'celeba/raw'
    processed_folder = 'celeba/processed32'

    def __init__(self, root=DATA_PATH, split='train', transform=None, sr_scale_factor=4):
        super(SuperResolutionCelebA32Dataset, self).__init__(root=root, split=split, transform=transform)

        assert isinstance(sr_scale_factor, int) and sr_scale_factor > 1
        self.sr_scale_factor = sr_scale_factor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """

        hr = self.data[index]

        if self.transform is not None:
            hr = self.transform(hr)

        lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        return (hr, lr)

    
class SuperResolutionCelebA64Dataset(UnsupervisedCelebA64Dataset):
    """
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    """
    raw_folder = 'celeba/raw'
    processed_folder = 'celeba/processed64'

    def __init__(self, root=DATA_PATH, split='train', transform=None, sr_scale_factor=4):
        super(UnsupervisedCelebA64Dataset, self).__init__(root=root, split=split, transform=transform)

        assert isinstance(sr_scale_factor, int) and sr_scale_factor > 1
        self.sr_scale_factor = sr_scale_factor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """

        hr = self.data[index]

        if self.transform is not None:
            hr = self.transform(hr)

        lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        return (hr, lr)


class SuperResolutionCelebA128Dataset(UnsupervisedCelebA128Dataset):
    """
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    """
    raw_folder = 'celeba/raw'
    processed_folder = 'celeba/processed128'

    def __init__(self, root=DATA_PATH, split='train', transform=None, sr_scale_factor=4):
        super(UnsupervisedCelebA128Dataset, self).__init__(root=root, split=split, transform=transform)

        assert isinstance(sr_scale_factor, int) and sr_scale_factor > 1
        self.sr_scale_factor = sr_scale_factor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """

        hr = self.data[index]

        if self.transform is not None:
            hr = self.transform(hr)

        lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        return (hr, lr)
