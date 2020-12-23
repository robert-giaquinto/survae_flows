import torch
from torchvision.datasets import CelebA
from survae.data import DATA_PATH


class SupervisedCelebADataset(CelebA):
    def __init__(self, root=DATA_PATH, split='train', transform=None, download=False):
        super(SupervisedCelebADataset, self).__init__(root,
                                                       split=split,
                                                       transform=transform,
                                                       target_type='attr',
                                                       download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(SupervisedCelebADataset, self).__getitem__(index)
