from survae.data.datasets.image import SuperResolutionCIFAR10Dataset, UnsupervisedCIFAR10Dataset, SupervisedCIFAR10Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class CIFAR10(TrainTestLoader):
    '''
    The CIFAR10 dataset of (Krizhevsky & Hinton, 2009):
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):

        self.root = root
        self.y_classes = 10
        self.sr_scale_factor = sr_scale_factor

        trans = [ToTensor(), Quantize(num_bits)]
        if resize_hw is not None:
            trans.insert(0, Resize((resize_hw, resize_hw)))
                         
        trans_train = pil_transforms + trans
        trans_test = trans

        # Load data
        if super_resolution:
            self.train = SuperResolutionCIFAR10Dataset(root, train=True, transform=Compose(trans_train), download=download, sr_scale_factor=sr_scale_factor)
            self.test = SuperResolutionCIFAR10Dataset(root, train=False, transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
        elif conditional:
            one_hot_encode = lambda target: F.one_hot(torch.tensor(target), self.y_classes)
            self.train = SupervisedCIFAR10Dataset(root, train=True, transform=Compose(trans_train), target_transform=one_hot_encode, download=download)
            self.test = SupervisedCIFAR10Dataset(root, train=False, transform=Compose(trans_test), target_transform=one_hot_encode)
        else:
            self.train = UnsupervisedCIFAR10Dataset(root, train=True, transform=Compose(trans_train), download=download)
            self.test = UnsupervisedCIFAR10Dataset(root, train=False, transform=Compose(trans_test))

