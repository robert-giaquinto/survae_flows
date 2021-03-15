import math
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
from survae.data.loaders.image import DynamicallyBinarizedMNIST, FixedBinarizedMNIST, MNIST
from survae.data.loaders.image import CIFAR10, ImageNet32, ImageNet64, SVHN, CelebA32, CelebA64

dataset_choices = {'cifar10', 'imagenet32', 'imagenet64', 'svhn', 'mnist', 'binary_mnist', 'celeba32', 'celeba64'}


def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='mnist', choices=dataset_choices)
    parser.add_argument('--num_bits', type=int, default=8)
    parser.add_argument('--sr_scale_factor', type=int, default=4,
                        help='Resizing factor for low-resolution image in super-resolution models (2=half, 4=quarter, etc.)')

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

    if args.super_resolution:
        cond_shape = get_sr_shape(args.dataset, args.sr_scale_factor)
    elif args.conditional:
        cond_shape = get_label_shape(args.dataset)
    else:
        cond_shape = None
    
    if args.dataset == 'binary_mnist':
        dataset = DynamicallyBinarizedMNIST()
    elif args.dataset == 'mnist':
        dataset = MNIST(num_bits=args.num_bits, pil_transforms=pil_transforms, conditional=args.conditional, super_resolution=args.super_resolution, sr_scale_factor=args.sr_scale_factor)
    elif args.dataset == 'cifar10':
        dataset = CIFAR10(num_bits=args.num_bits, pil_transforms=pil_transforms, conditional=args.conditional, super_resolution=args.super_resolution, sr_scale_factor=args.sr_scale_factor)
    elif args.dataset == 'imagenet32':
        dataset = ImageNet32(num_bits=args.num_bits, pil_transforms=pil_transforms, conditional=args.conditional, super_resolution=args.super_resolution, sr_scale_factor=args.sr_scale_factor)
    elif args.dataset == 'imagenet64':
        dataset = ImageNet64(num_bits=args.num_bits, pil_transforms=pil_transforms, conditional=args.conditional, super_resolution=args.super_resolution, sr_scale_factor=args.sr_scale_factor)
    elif args.dataset == 'svhn':
        dataset = SVHN(num_bits=args.num_bits, pil_transforms=pil_transforms, conditional=args.conditional, super_resolution=args.super_resolution, sr_scale_factor=args.sr_scale_factor)
    elif args.dataset == 'celeba32':
        dataset = CelebA32(num_bits=args.num_bits, pil_transforms=pil_transforms, conditional=args.conditional, super_resolution=args.super_resolution, sr_scale_factor=args.sr_scale_factor)
    elif args.dataset == 'celeba64':
        dataset = CelebA64(num_bits=args.num_bits, pil_transforms=pil_transforms, conditional=args.conditional, super_resolution=args.super_resolution, sr_scale_factor=args.sr_scale_factor)
    else:
        raise ValueError(f"{dataset} is an unrecognized dataset.")

    # Data Loader
    loaders = dataset.get_data_loaders(batch_size=args.batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers)
    train_loader, test_loader = loaders[0], loaders[-1]

    return train_loader, test_loader, data_shape, cond_shape


def get_augmentation(augmentation, dataset, data_shape):
    c, h, w = data_shape
    if augmentation is None or augmentation == "none":
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
        data_shape = (1,28,28)
    elif dataset == 'cifar10':
        data_shape = (3,32,32)
    elif dataset == 'imagenet32':
        data_shape = (3,32,32)
    elif dataset == 'imagenet64':
        data_shape = (3,64,64)
    elif dataset == 'svhn':
        data_shape = (3,32,32)
    elif dataset == 'celeba32':
        data_shape = (3,32,32)
    elif dataset == 'celeba64':
        data_shape = (3,64,64)
    else:
        raise ValueError(f"{dataset} is an unrecognized dataset.")
        
    return data_shape


def get_sr_shape(dataset, sr_scale_factor):
    data_shape = get_data_shape(dataset)
    assert data_shape[1] % sr_scale_factor == 0 and data_shape[2] % sr_scale_factor == 0
    sr_shape = (data_shape[0], data_shape[1] // sr_scale_factor, data_shape[2] // sr_scale_factor)
    return sr_shape


def get_label_shape(dataset):
    """
    Just a placehold for now, conditional models should be up and running soon!
    """
    if dataset in ['mnist', 'binary_mnist']:
        cond_shape = (1,)
    elif dataset == 'cifar10':
        cond_shape = None
    elif dataset == 'imagenet32':
        cond_shape = None
    elif dataset == 'imagenet64':
        cond_shape = None
    elif dataset == 'svhn':
        cond_shape = None
    elif dataset == 'celeba32':
        cond_shape = None
    elif dataset == 'celeba32':
        cond_shape = None
    else:
        raise ValueError(f"{dataset} is an unrecognized dataset.")


    return cond_shape


