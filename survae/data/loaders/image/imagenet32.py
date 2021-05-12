from survae.data.datasets.image import UnsupervisedImageNet32Dataset, SuperResolutionImageNet32Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from survae.data.transforms import Quantize
from survae.data import TrainTestLoader, DATA_PATH


class ImageNet32(TrainTestLoader):
    '''
    The ImageNet dataset of
    (Russakovsky et al., 2015): https://arxiv.org/abs/1409.0575
    downscaled to 32x32, as used in
    (van den Oord et al., 2016): https://arxiv.org/abs/1601.06759
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):

        self.root = root

        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]
        if resize_hw is not None:
            trans_train.insert(0, Resize((resize_hw, resize_hw)))
            trans_test.insert(0, Resize((resize_hw, resize_hw)))

        # Load data
        if super_resolution:
            self.train = SuperResolutionImageNet32Dataset(root, train=True, transform=Compose(trans_train), download=download, sr_scale_factor=sr_scale_factor)
            self.test = SuperResolutionImageNet32Dataset(root, train=False, transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
        else:
            self.train = UnsupervisedImageNet32Dataset(root, train=True, transform=Compose(trans_train), download=download)
            self.test = UnsupervisedImageNet32Dataset(root, train=False, transform=Compose(trans_test))
