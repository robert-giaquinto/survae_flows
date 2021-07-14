import os
import torch
from torchvision.datasets import MNIST
from torchvision.transforms.functional import resize
from torchvision.transforms import Compose, ToTensor
from survae.data import DATA_PATH


class SuperResolutionMNISTDataset(MNIST):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False, sr_scale_factor=4, bicubic=False):

        if transform is None:
            transform = Compose([ToTensor()])
        
        super(SuperResolutionMNISTDataset, self).__init__(root,
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
            tuple: (high-resolution image, low-resolution image)
        """
        hr, _ = super(SuperResolutionMNISTDataset, self).__getitem__(index)

        if self.bicubic:
            lr = resize(hr, hr.shape[0] // self.sr_scale_factor, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            # h_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # w_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # lr = hr[:, h_offset::self.sr_scale_factor, w_offset::self.sr_scale_factor]
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
