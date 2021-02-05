import math
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
from survae.data.loaders.image import DynamicallyBinarizedMNIST, FixedBinarizedMNIST, MNIST
from survae.data.loaders.image import CIFAR10, ImageNet32, ImageNet64, SVHN

dataset_choices = {'cifar10', 'imagenet32', 'imagenet64', 'svhn', 'mnist', 'binary_mnist'}


def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='mnist', choices=dataset_choices)
    parser.add_argument('--num_bits', type=int, default=8)

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=True)
    parser.add_argument('--augmentation', type=str, default=None)



def get_data_id(args):
    return '{}_{}bit'.format(args.dataset, args.num_bits)


def get_data(args):
    assert args.dataset in dataset_choices

    data_shape = get_data_shape(args.dataset)
    pil_transforms = get_augmentation(args.augmentation, args.dataset, data_shape)
    
    if args.dataset == 'binary_mnist':
        dataset = DynamicallyBinarizedMNIST()
    elif args.dataset == 'mnist':
        dataset = MNIST(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == 'cifar10':
        dataset = CIFAR10(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == 'imagenet32':
        dataset = ImageNet32(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == 'imagenet64':
        dataset = ImageNet64(num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif args.dataset == 'svhn':
        dataset = SVHN(num_bits=args.num_bits, pil_transforms=pil_transforms)


    # Data Loader
    train_loader, test_loader = dataset.get_data_loaders(args.batch_size)

    return train_loader, test_loader, data_shape


def get_augmentation(augmentation, dataset, data_shape):
    c, h, w = data_shape
    if augmentation is None:
        pil_transforms = []
    elif augmentation == 'horizontal_flip':
        pil_transforms = [RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'neta':
        assert h==w
        pil_transforms = [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    elif augmentation == 'eta':
        assert h==w
        pil_transforms = [RandomHorizontalFlip(),
                          Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    return pil_transforms


def get_data_shape(dataset):
    if dataset in ['mnist', 'binary_mnist']:
        return (1,28,28)
    elif dataset == 'cifar10':
        return (3,32,32)
    elif dataset == 'imagenet32':
        return (3,32,32)
    elif dataset == 'imagenet64':
        return (3,64,64)
    elif dataset == 'svhn':
        return (3,32,32)

