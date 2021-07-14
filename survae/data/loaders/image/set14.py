from survae.data.datasets.image import Set14Dataset
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop, CenterCrop
from survae.data.transforms import Quantize
from survae.data import TrainValidTestLoader, DATA_PATH


class Set14(TrainValidTestLoader):
    def __init__(self, resize_hw, root=DATA_PATH, num_bits=8, pil_transforms=[], sr_scale_factor=4, bicubic=False, crop=None, repeats=1):
        super(Set14, self).__init__()
        self.root = root

        if crop is not None:
            if crop == "random":
                trans_test = Compose([RandomCrop(resize_hw), ToTensor(), Quantize(num_bits)])
            elif crop == "center":
                trans_test = Compose([CenterCrop(resize_hw), ToTensor(), Quantize(num_bits)])
            else:
                raise ValueError("crop must be None, 'random', or 'center'")
                                 
            self.test = Set14Dataset(resize_hw=None, root=root, split='test', transform=trans_test, sr_scale_factor=sr_scale_factor, bicubic=bicubic, repeats=repeats)
        else:
            trans_test = Compose([ToTensor(), Quantize(num_bits)])
            self.test = Set14Dataset(resize_hw=resize_hw, root=root, split='test', transform=trans_test, sr_scale_factor=sr_scale_factor, bicubic=bicubic, repeats=repeats)

















































