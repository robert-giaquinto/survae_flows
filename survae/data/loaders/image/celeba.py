from survae.data.datasets.image import SuperResolutionCelebADataset, UnsupervisedCelebADataset, SupervisedCelebADataset
from torchvision.transforms import Compose, ToTensor, Resize
from survae.data.transforms import Quantize
from survae.data import TrainValidTestLoader, DATA_PATH


class CelebA32(TrainValidTestLoader):
    '''
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    preprocessed to 64x64 as in
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    '''

    def __init__(self, root=DATA_PATH, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4):

        self.root = root
        self.input_size = [3, 32, 32]
        self.y_classes = 40

        trans_train = pil_transforms + [Resize(self.input_size[1:]), ToTensor(), Quantize(num_bits)]
        trans_test = [Resize(self.input_size[1:]), ToTensor(), Quantize(num_bits)]        

        # Load data
        if super_resolution:
            self.train = SuperResolutionCelebADataset(root, split='train', transform=Compose(trans_train), sr_scale_factor=sr_scale_factor)
            self.valid = SuperResolutionCelebADataset(root, split='valid', transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
            self.test = SuperResolutionCelebADataset(root, split='test', transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
        elif conditional:
            one_hot_encode = lambda target: F.one_hot(torch.tensor(target), self.y_classes)  # needed?
            self.train = SupervisedCelebADataset(root, split='train', transform=Compose(trans_train), target_transform=one_hot_encode)
            self.valid = SupervisedCelebADataset(root, split='valid', transform=Compose(trans_test), target_transform=one_hot_encode)
            self.test = SupervisedCelebADataset(root, split='test', transform=Compose(trans_test), target_transform=one_hot_encode)
        else:
            self.train = UnsupervisedCelebADataset(root, split='train', transform=Compose(trans_train))
            self.valid = UnsupervisedCelebADataset(root, split='valid', transform=Compose(trans_test))
            self.test = UnsupervisedCelebADataset(root, split='test', transform=Compose(trans_test))


class CelebA64(TrainValidTestLoader):
    '''
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    preprocessed to 64x64 as in
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    '''

    def __init__(self, root=DATA_PATH, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4):

        self.root = root
        self.input_size = [3, 64, 64]
        self.y_classes = 40

        trans_train = pil_transforms + [Resize(self.input_size[1:]), ToTensor(), Quantize(num_bits)]
        trans_test = [Resize(self.input_size[1:]), ToTensor(), Quantize(num_bits)]

        # Load data
        if super_resolution:
            self.train = SuperResolutionCelebADataset(root, split='train', transform=Compose(trans_train), sr_scale_factor=sr_scale_factor)
            self.valid = SuperResolutionCelebADataset(root, split='valid', transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
            self.test = SuperResolutionCelebADataset(root, split='test', transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
        elif conditional:
            one_hot_encode = lambda target: F.one_hot(torch.tensor(target), self.y_classes)  # needed?
            self.train = SupervisedCelebADataset(root, split='train', transform=Compose(trans_train), target_transform=one_hot_encode)
            self.valid = SupervisedCelebADataset(root, split='valid', transform=Compose(trans_test), target_transform=one_hot_encode)
            self.test = SupervisedCelebADataset(root, split='test', transform=Compose(trans_test), target_transform=one_hot_encode)
        else:
            self.train = UnsupervisedCelebADataset(root, split='train', transform=Compose(trans_train))
            self.valid = UnsupervisedCelebADataset(root, split='valid', transform=Compose(trans_test))
            self.test = UnsupervisedCelebADataset(root, split='test', transform=Compose(trans_test))

