import torch
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, ToTensor
from survae.data import DATA_PATH


class SuperResolutionSVHNDataset(SVHN):
    def __init__(self, root=DATA_PATH, split='train', transform=None, download=False, sr_scale_factor=4):

        if transform is None:
            transform = Compose([ToTensor()])

        super(SuperResolutionSVHNDataset, self).__init__(root,
                                                         split=split,
                                                         transform=transform,
                                                         download=download)
        assert isinstance(sr_scale_factor, int) and sr_scale_factor > 1
        self.sr_scale_factor = sr_scale_factor


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        hr, _ = super(SuperResolutionSVHNDataset, self).__getitem__(index)
        lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        return (hr, lr)
