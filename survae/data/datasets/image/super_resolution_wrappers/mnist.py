import os
import torch
from torchvision.datasets import MNIST
import torch.nn.functional as F
from survae.data import DATA_PATH


class SuperResolutionMNISTDataset(MNIST):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False, sr_scale_factor=4):
        super(SuperResolutionMNISTDataset, self).__init__(root,
                                              train=train,
                                              transform=transform,
                                              download=download)
        assert isinstance(sr_scale_factor, int) and sr_scale_factor > 1
        self.sr_scale_factor = sr_scale_factor

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (high-resolution image, low-resolution image)
        """
        hr, _ = super(SuperResolutionMNISTDataset, self).__getitem__(index)
        # use interpolate to resize data since the data is already in tensor form (not images)
        #lr = F.interpolate(hr, size=(14,14))
        lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        return (hr, lr)

    @property
    def raw_folder(self):
        # Replace self.__class__.__name__ by 'MNIST'
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        # Replace self.__class__.__name__ by 'MNIST'
        return os.path.join(self.root, 'MNIST', 'processed')
