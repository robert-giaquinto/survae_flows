import torch
from torchvision.datasets import SVHN
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import resize
from survae.data import DATA_PATH


class SuperResolutionSVHNDataset(SVHN):
    def __init__(self, root=DATA_PATH, split='train', transform=None, download=False, sr_scale_factor=4, bicubic=False):

        if transform is None:
            transform = Compose([ToTensor()])

        super(SuperResolutionSVHNDataset, self).__init__(root,
                                                         split=split,
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
        hr, _ = super(SuperResolutionSVHNDataset, self).__getitem__(index)

        if self.bicubic:
            lr = resize(hr, hr.shape[0] // self.sr_scale_factor, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            # h_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # w_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # lr = hr[:, h_offset::self.sr_scale_factor, w_offset::self.sr_scale_factor]
            lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        
        return (hr, lr)
