import os
import torch
import torch.utils.data as data
import numpy as np
import errno
import tarfile
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import resize

from survae.data import DATA_PATH
from survae.data.datasets.image.unsupervised_wrappers.imagenet32 import UnsupervisedImageNet32Dataset


class SuperResolutionImageNet32Dataset(UnsupervisedImageNet32Dataset):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False, sr_scale_factor=4, bicubic=False):

        if transform is None:
            transform = Compose([ToTensor()])

        super(SuperResolutionImageNet32Dataset, self).__init__(root,
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
            tuple: (image, target) where target is index of the target class.
        """
        hr = super(SuperResolutionImageNet32Dataset, self).__getitem__(index)
        
        if self.bicubic:
            lr = resize(hr, hr.shape[0] // self.sr_scale_factor, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            # h_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # w_offset = torch.randint(self.sr_scale_factor, (1,)).item()
            # lr = hr[:, h_offset::self.sr_scale_factor, w_offset::self.sr_scale_factor]
            lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
            
        return (hr, lr)


class _SuperResolutionImageNet32Dataset(data.Dataset):
    """
    The ImageNet dataset of
    (Russakovsky et al., 2015): https://arxiv.org/abs/1409.0575
    downscaled to 32x32, as used in
    (van den Oord et al., 2016): https://arxiv.org/abs/1601.06759
    """

    urls = [
        'http://image-net.org/small/train_32x32.tar',
        'http://image-net.org/small/valid_32x32.tar'
    ]
    raw_folder = 'imagenet32/raw'
    processed_folder = 'imagenet32/processed'
    train_folder = 'train_32x32'
    valid_folder = 'valid_32x32'

    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False, sr_scale_factor=4):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        assert isinstance(sr_scale_factor, int) and sr_scale_factor > 1
        self.sr_scale_factor = sr_scale_factor

        if not self._check_raw():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' +
                                   ' You can use download=True to download it')

        if not self._check_processed():
            self.process()

        if self.train:
            self.files = [os.path.join(self.processed_train_folder, file) for file in os.listdir(self.processed_train_folder)]
        else:
            self.files = [os.path.join(self.processed_valid_folder, file) for file in os.listdir(self.processed_valid_folder)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """
        hr = Image.open(self.files[index])

        if self.transform is not None:
            hr = self.transform(hr)

        lr = hr[:, ::self.sr_scale_factor, ::self.sr_scale_factor]
        return (hr, lr)

    def __len__(self):
        return len(self.files)

    @property
    def raw_file_paths(self):
        return [os.path.join(self.root, self.raw_folder, url.rpartition('/')[2]) for url in self.urls]

    @property
    def processed_data_folder(self):
        return os.path.join(self.root, self.processed_folder)

    @property
    def processed_train_folder(self):
        return os.path.join(self.processed_data_folder, self.train_folder)

    @property
    def processed_valid_folder(self):
        return os.path.join(self.processed_data_folder, self.valid_folder)

    def _check_processed(self):
        return os.path.exists(self.processed_train_folder) and os.path.exists(self.processed_valid_folder)

    def _check_raw(self):
        return os.path.exists(self.raw_file_paths[0]) and os.path.exists(self.raw_file_paths[1])

    def download(self):
        """Download the data if it doesn't exist in processed_folder already."""
        from six.moves import urllib

        if self._check_raw():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url, file_path in zip(self.urls, self.raw_file_paths):
            print('Downloading ' + url)
            urllib.request.urlretrieve(url, file_path)

    def process(self):

        print("Extracting training data...")
        tar = tarfile.open(self.raw_file_paths[0])
        tar.extractall(self.processed_data_folder)
        tar.close()

        print("Extracting validation data...")
        tar = tarfile.open(self.raw_file_paths[1])
        tar.extractall(self.processed_data_folder)
        tar.close()
