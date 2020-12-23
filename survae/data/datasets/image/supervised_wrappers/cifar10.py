import torch
from torchvision.datasets import CIFAR10
from survae.data import DATA_PATH


class SupervisedCIFAR10Dataset(CIFAR10):
    def __init__(self, root=DATA_PATH, train=True, transform=None, target_transform=None, download=False):
        super(SupervisedCIFAR10Dataset, self).__init__(root,
                                                 train=train,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(SupervisedCIFAR10Dataset, self).__getitem__(index)
