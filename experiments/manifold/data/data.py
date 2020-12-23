import math
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
from survae.data.loaders.image import DynamicallyBinarizedMNIST

dataset_choices = {'mnist'}


def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='mnist', choices=dataset_choices)
    parser.add_argument('--num_bits', type=int, default=8)

    # Train params
    parser.add_argument('--batch_size', type=int, default=128)


def get_data_id(args):
    return '{}_{}bit'.format(args.dataset, args.num_bits)


def get_data(args):
    assert args.dataset in dataset_choices

    data_shape = get_data_shape(args.dataset)
    if args.dataset == 'mnist':
        dataset = DynamicallyBinarizedMNIST()

    # Data Loader
    train_loader, test_loader = dataset.get_data_loaders(args.batch_size)

    return train_loader, test_loader, data_shape


def get_data_shape(dataset):
    if dataset == 'mnist':
        return (1,28,28)
