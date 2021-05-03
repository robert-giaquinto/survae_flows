import os
import torch
import torch.utils.data as data
import numpy as np
import errno

import zipfile
import pickle

from PIL import Image
from survae.data import DATA_PATH


class UnsupervisedImageNet64Dataset(data.Dataset):
    """
    The ImageNet dataset of
    (Russakovsky et al., 2015): https://arxiv.org/abs/1409.0575
    downscaled to 64x64, as used in
    (van den Oord et al., 2016): https://arxiv.org/abs/1601.06759

    OLD:
    urls = [
        'http://image-net.org/small/train_64x64.tar',
        'http://image-net.org/small/valid_64x64.tar'
    ]

    urls = [
        'https://image-net.org/data/downsample/Imagenet64_train_part1_npz.zip',
        'https://image-net.org/data/downsample/Imagenet64_train_part2_npz.zip',
        'https://image-net.org/data/downsample/Imagenet64_val_npz.zip'
    ]
    """

    urls = [
        'https://image-net.org/data/downsample/Imagenet64_train_part1.zip',
        'https://image-net.org/data/downsample/Imagenet64_train_part2.zip',
        'https://image-net.org/data/downsample/Imagenet64_val.zip'
    ]

    train_list = [
        'train_data_batch_1',
        'train_data_batch_2',
        'train_data_batch_3',
        'train_data_batch_4',
        'train_data_batch_5',
        'train_data_batch_6',
        'train_data_batch_7',
        'train_data_batch_8',
        'train_data_batch_9',
        'train_data_batch_10'
    ]
    test_list = ['val_data']
    
    raw_folder = 'imagenet64/raw'
    processed_folder = 'imagenet64/processed'
    train_folder = 'train_64x64'
    valid_folder = 'valid_64x64'

    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        if not self._check_raw():
            if download:
                self.download()
            else:
                raise RuntimeError(f'Datasets  not found. You can use download=True to download: {self.raw_file_paths}')

        if not self._check_processed():
            self.process()
        
        # now load the picked numpy arrays
        if self.train:
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                with open(os.path.join(self.processed_train_folder, fentry), 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                    self.data.append(entry['data'])
                    # if 'labels' in entry:
                    #     self.labels += entry['labels']
                    # else:
                    #     self.labels += entry['fine_labels']

            #self.labels[:] = [x - 1 for x in self.labels]  # resize label range from [1,1000] to [0,1000)
            self.data = np.concatenate(self.data)
            [picnum, pixel] = self.data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.data = self.data.reshape((picnum, 3, pixel, pixel))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            with open(os.path.join(self.processed_valid_folder, self.test_list[0]), 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.data = entry['data']
                [picnum, pixel]= self.data.shape
                pixel = int(np.sqrt(pixel/3))
                # if 'labels' in entry:
                #     self.labels = entry['labels']
                # else:
                #     self.labels = entry['fine_labels']

            #self.labels[:] = [x - 1 for x in self.labels]  # resize label range from [1,1000] to [0,1000)
            self.data = self.data.reshape((picnum, 3, pixel, pixel))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        # target = self.labels[index]
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img

    def __len__(self):
        return len(self.data)

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
        return os.path.exists(self.raw_file_paths[0]) and os.path.exists(self.raw_file_paths[1]) and os.path.exists(self.raw_file_paths[2])

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

    def process_file(fname, out_dir):
        print(f"Extracting data from {fname} into {out_dir}. Flattening zipped folder structure.")
        with zipfile.ZipFile(fname) as zip_file:
            for zip_info in zip_file.infolist():
                # if zip_info.filename[-1] == '/':
                #     continue

                # zip_info.filename = os.path.basename(zip_info.filename)
                zip_file.extract(zip_info, out_dir)


    def process(self):
        self.process_file(self.raw_file_paths[0], self.processed_train_folder)
        self.process_file(self.raw_file_paths[1], self.processed_train_folder)
        self.process_file(self.raw_file_paths[2], self.processed_valid_folder)
