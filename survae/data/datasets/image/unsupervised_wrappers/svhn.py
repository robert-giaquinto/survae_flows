import torch
from torchvision.datasets import SVHN
from survae.data import DATA_PATH


class UnsupervisedSVHNDataset(SVHN):
    def __init__(self, root=DATA_PATH, split='train', transform=None, download=False):
        super(UnsupervisedSVHNDataset, self).__init__(root,
                                               split=split,
                                               transform=transform,
                                               download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(UnsupervisedSVHNDataset, self).__getitem__(index)[0]
