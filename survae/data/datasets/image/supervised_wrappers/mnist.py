import os
import torch
from torchvision.datasets import MNIST
from survae.data import DATA_PATH


class SupervisedMNISTDataset(MNIST):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        super(SupervisedMNISTDataset, self).__init__(root,
                                              train=train,
                                              transform=transform,
                                              download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #return super(SupervisedMNISTDataset, self).__getitem__(index)
        x, y = super(SupervisedMNISTDataset, self).__getitem__(index)
        y = torch.tensor(y).view(1,1,1) * 1.0
        return (x, y)

    @property
    def raw_folder(self):
        # Replace self.__class__.__name__ by 'MNIST'
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        # Replace self.__class__.__name__ by 'MNIST'
        return os.path.join(self.root, 'MNIST', 'processed')
