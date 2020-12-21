from survae.data.datasets.image import UnsupervisedMNISTDataset, SupervisedMNISTDataset, SuperResolutionMNISTDataset
from torchvision.transforms import Compose, ToTensor
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class MNIST(TrainTestLoader):
    '''
    The MNIST dataset of (LeCun, 1998):
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False):

        self.root = root
        self.y_classes = 10

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]

        # Load data
        if super_resolution:
            self.train = SuperResolutionMNISTDataset(root, train=True, transform=Compose(trans_train), download=download)
            self.test = SuperResolutionMNISTDataset(root, train=False, transform=Compose(trans_test))
        elif conditional:
            self.train = SupervisedMNISTDataset(root, train=True, transform=Compose(trans_train), download=download)
            self.test = SupervisedMNISTDataset(root, train=False, transform=Compose(trans_test))
        else:
            self.train = UnsupervisedMNISTDataset(root, train=True, transform=Compose(trans_train), download=download)
            self.test = UnsupervisedMNISTDataset(root, train=False, transform=Compose(trans_test))

