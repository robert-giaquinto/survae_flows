from survae.data.datasets.image import UnsupervisedMNISTDataset, SupervisedMNISTDataset, SuperResolutionMNISTDataset
from torchvision.transforms import Compose, ToTensor, Resize
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class MNIST(TrainTestLoader):
    '''
    The MNIST dataset of (LeCun, 1998):
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):

        self.root = root
        self.y_classes = 10
        self.sr_scale_factor = sr_scale_factor

        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]
        if resize_hw is not None:
            trans_train.insert(0, Resize((resize_hw, resize_hw)))
            trans_test.insert(0, Resize((resize_hw, resize_hw)))

        # Load data
        if super_resolution:
            self.train = SuperResolutionMNISTDataset(root, train=True, transform=Compose(trans_train), download=download, sr_scale_factor=sr_scale_factor)
            self.test = SuperResolutionMNISTDataset(root, train=False, transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
        elif conditional:
            self.train = SupervisedMNISTDataset(root, train=True, transform=Compose(trans_train), download=download)
            self.test = SupervisedMNISTDataset(root, train=False, transform=Compose(trans_test))
        else:
            self.train = UnsupervisedMNISTDataset(root, train=True, transform=Compose(trans_train), download=download)
            self.test = UnsupervisedMNISTDataset(root, train=False, transform=Compose(trans_test))

