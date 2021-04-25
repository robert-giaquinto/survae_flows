from torch.utils.data import DataLoader
from survae.data.loaders.toy import Gaussian, Crescent, CrescentCubed, SineWave, Abs, Sign, FourCircles, Diamond, TwoSpirals, TwoMoons, Checkerboard, Face

from data.datasets import CornersDataset, EightGaussiansDataset

dataset_choices = {'gaussian', 'crescent', 'crescent_cubed', 'sinewave', 'abs', 'sign', 'four_circles', 'diamond', 'two_spirals', 'two_moons', 'checkerboard', 'face_einstein', 'corners', 'eight_gaussians'}


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='checkerboard', choices=dataset_choices)
    parser.add_argument('--train_samples', type=int, default=128*1000)
    parser.add_argument('--test_samples', type=int, default=128*1000)

    # Train params
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    

def get_data_id(args):
    return '{}'.format(args.dataset)


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    if args.dataset == 'gaussian':
        dataset = Gaussian(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'crescent':
        dataset = Crescent(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'crescent_cubed':
        dataset = CrescentCubed(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'sinewave':
        dataset = SineWave(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'abs':
        dataset = Abs(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'sign':
        dataset = Sign(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'diamond':
        dataset = Diamond(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'four_circles':
        dataset = FourCircles(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'two_spirals':
        dataset = TwoSpirals(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'two_moons':
        dataset = TwoMoons(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'checkerboard':
        dataset = Checkerboard(train_samples=args.train_samples, test_samples=args.test_samples)
    elif args.dataset == 'face_einstein':
        dataset = Face(train_samples=args.train_samples, test_samples=args.test_samples, name='einstein')
    elif args.dataset == 'corners':
        dataset = DataContainer(CornersDataset(args.train_samples), CornersDataset(args.test_samples))
    elif args.dataset == 'eight_gaussians':
        dataset = DataContainer(EightGaussiansDataset(args.train_samples), EightGaussiansDataset(args.test_samples))

    # Data Loader
    train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset.test, batch_size=args.batch_size, shuffle=False)

    return train_loader, eval_loader


class DataContainer():
    def __init__(self, train, test):
        self.train = train
        self.test = test
