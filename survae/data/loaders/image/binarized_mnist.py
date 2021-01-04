from survae.data.datasets.image import UnsupervisedMNISTDataset, SupervisedMNISTDataset
from torchvision.transforms import Compose, ToTensor #, ConvertImageDtype
from survae.data.transforms import Flatten, DynamicBinarize
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

    def __init__(self, root=DATA_PATH, download=True, flatten=False, as_float=False, conditional=False, super_resolution=False):

        self.root = root
        self.y_classes = 10

        # Define transformations
        trans = [ToTensor(), DynamicBinarize(as_float=as_float)]
        #if as_float: trans.append(ConvertImageDtype(torch.float))
        if flatten: trans.append(Flatten())

        # Load data
        if conditional:
            self.train = SupervisedMNISTDataset(root, train=True, transform=Compose(trans), download=download)
            self.test = SupervisedMNISTDataset(root, train=False, transform=Compose(trans))
        else:
            self.train = UnsupervisedMNISTDataset(root, train=True, transform=Compose(trans), download=download)
            self.test = UnsupervisedMNISTDataset(root, train=False, transform=Compose(trans))
