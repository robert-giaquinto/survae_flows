import os
from survae.data.datasets.image import UnsupervisedSVHNDataset, SupervisedSVHNDataset, SuperResolutionSVHNDataset
from torchvision.transforms import Compose, ToTensor, Resize
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class SVHN(TrainTestLoader):
    '''
    The SVHN dataset of (Netzer et al., 2011):
    https://research.google/pubs/pub37648/
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):

        self.root = root
        self.sr_scale_factor = sr_scale_factor

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]
        if resize_hw is not None:
            trans_train.insert(0, Resize((resize_hw, resize_hw)))
            trans_test.insert(0, Resize((resize_hw, resize_hw)))
                         
        # Load data
        sub_root = os.path.join(root, 'SVHN')

        if super_resolution:
            self.train = SuperResolutionSVHNDataset(sub_root, split='train', transform=Compose(trans_train), download=download, sr_scale_factor=sr_scale_factor)
            self.test = SuperResolutionSVHNDataset(sub_root, split='test', transform=Compose(trans_test), download=download, sr_scale_factor=sr_scale_factor)
        elif conditional:
            self.train = SupervisedSVHNDataset(sub_root, split='train', transform=Compose(trans_train), download=download)
            self.test = SupervisedSVHNDataset(sub_root, split='test', transform=Compose(trans_test), download=download)
        else:
            self.train = UnsupervisedSVHNDataset(sub_root, split='train', transform=Compose(trans_train), download=download)
            self.test = UnsupervisedSVHNDataset(sub_root, split='test', transform=Compose(trans_test), download=download)
