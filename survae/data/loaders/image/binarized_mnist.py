from survae.data.datasets.image import UnsupervisedMNISTDataset, SupervisedMNISTDataset, SuperResolutionMNISTDataset
from survae.data.transforms import Flatten, DynamicBinarize
from torchvision.transforms import Compose, ToTensor, Resize
from survae.data import TrainTestLoader, DATA_PATH


class DynamicallyBinarizedMNIST(TrainTestLoader):
    '''
    The MNIST dataset of (LeCun, 1998):
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    with a dynamic binarization as used in (Salakhutdinov & Murray, 2008):
    https://www.cs.toronto.edu/~rsalakhu/papers/dbn_ais.pdf

    See Footnote 2 on page 6 and Appendix D of (Burda et al., 2015):
    https://arxiv.org/pdf/1509.00519.pdf
    for a remark on the different versions of MNIST.
    '''

    def __init__(self, root=DATA_PATH, download=True, flatten=False, as_float=False, conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):

        self.root = root
        self.y_classes = 10
        self.sr_scale_factor = sr_scale_factor
        
        trans_train = pil_transforms + [ToTensor(), DynamicBinarize(as_float=as_float)]
        trans_test = [ToTensor(), DynamicBinarize(as_float=as_float)]
        if resize_hw is not None:
            trans_train.insert(0, Resize((resize_hw, resize_hw)))
            trans_test.insert(0, Resize((resize_hw, resize_hw)))

        if flatten:
            trans.append(Flatten())

        # Load data
        if super_resolution:
            self.train = SuperResolutionMNISTDataset(root, train=True, transform=Compose(trans_train), download=download, sr_scale_factor=sr_scale_factor)
            self.test = SuperResolutionMNISTDataset(root, train=False, transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
        elif conditional:
            self.train = SupervisedMNISTDataset(root, train=True, transform=Compose(trans), download=download)
            self.test = SupervisedMNISTDataset(root, train=False, transform=Compose(trans))
        else:
            self.train = UnsupervisedMNISTDataset(root, train=True, transform=Compose(trans), download=download)
            self.test = UnsupervisedMNISTDataset(root, train=False, transform=Compose(trans))
