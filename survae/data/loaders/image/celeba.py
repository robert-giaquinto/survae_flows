from survae.data.datasets.image import SuperResolutionCelebA32Dataset, UnsupervisedCelebA32Dataset
from survae.data.datasets.image import SuperResolutionCelebA64Dataset, UnsupervisedCelebA64Dataset
from survae.data.datasets.image import SuperResolutionCelebA128Dataset, UnsupervisedCelebA128Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from survae.data.transforms import Quantize
from survae.data import TrainValidTestLoader, DATA_PATH


class CelebA(TrainValidTestLoader):
    '''
    The CelebA dataset of
    (Liu et al., 2015): https://arxiv.org/abs/1411.7766
    preprocessed to 64x64 as in
    (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
    (Dinh et al., 2017): https://arxiv.org/abs/1605.08803
    '''
    def __init__(self, input_size, root=DATA_PATH, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):
        super(CelebA, self).__init__()

        assert len(input_size) == 3
        
        self.root = root
        self.input_size = input_size
        self.y_classes = 40

        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]
        if resize_hw is not None:
            trans_train.insert(0, Resize((resize_hw, resize_hw)))
            trans_test.insert(0, Resize((resize_hw, resize_hw)))

        if conditional:
            raise ValueError(f"Conditional CelebA dataset not available yet.")

        # Load data
        if super_resolution:
            if input_size[-1] == 32:
                Dataset = SuperResolutionCelebA32Dataset
            elif input_size[-1] == 64:
                Dataset = SuperResolutionCelebA64Dataset
            elif input_size[-1] == 128:
                Dataset = SuperResolutionCelebA128Dataset
            else:
                raise ValueError(f"Invalid input size {input_size}")

            self.train = Dataset(root, split='train', transform=Compose(trans_train), sr_scale_factor=sr_scale_factor)
            self.valid = Dataset(root, split='valid', transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
            self.test = Dataset(root, split='test', transform=Compose(trans_test), sr_scale_factor=sr_scale_factor)
            
        else:
            if input_size[-1] == 32:
                Dataset = UnsupervisedCelebA32Dataset
            elif input_size[-1] == 64:
                Dataset = UnsupervisedCelebA64Dataset
            elif input_size[-1] == 128:
                Dataset = UnsupervisedCelebA128Dataset
            else:
                raise ValueError(f"Invalid input size {input_size}")


            self.train = Dataset(root, split='train', transform=Compose(trans_train))
            self.valid = Dataset(root, split='valid', transform=Compose(trans_test))
            self.test = Dataset(root, split='test', transform=Compose(trans_test))


class CelebA32(CelebA):
    def __init__(self, root=DATA_PATH, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):
        super(CelebA32, self).__init__(input_size=[3,32,32], root=root,
                                       num_bits=num_bits, pil_transforms=pil_transforms,
                                       conditional=conditional, super_resolution=super_resolution, sr_scale_factor=sr_scale_factor, resize_hw=resize_hw)

class CelebA64(CelebA):
    def __init__(self, root=DATA_PATH, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):
        super(CelebA64, self).__init__(input_size=[3,64,64], root=root,
                                       num_bits=num_bits, pil_transforms=pil_transforms,
                                       conditional=conditional, super_resolution=super_resolution, sr_scale_factor=sr_scale_factor, resize_hw=resize_hw)

class CelebA128(CelebA):
    def __init__(self, root=DATA_PATH, num_bits=8, pil_transforms=[], conditional=False, super_resolution=False, sr_scale_factor=4, resize_hw=None):
        super(CelebA128, self).__init__(input_size=[3,128,128], root=root,
                                       num_bits=num_bits, pil_transforms=pil_transforms,
                                       conditional=conditional, super_resolution=super_resolution, sr_scale_factor=sr_scale_factor, resize_hw=resize_hw)


