import torch
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as F
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

        coin_flip = torch.randint(2, (1,)).item()
        if coin_flip > 0:
            lr = F.resize(hr, hr.shape[0] // self.sr_scale_factor)
        else:
            h_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            w_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            lr = hr[:, h_offset::self.sr_scale_factor, w_offset::self.sr_scale_factor]
        
        return (hr, lr)
