import argparse
import pickle
import numpy as np
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from model.unconditional_flow import UnconditionalFlow
from model.concrete_dropout import DropoutNet


def add_baseline_args(parser):

    # Model params
    parser.add_argument('--baseline', type=str, choices=["gp", "dropout"])
    parser.add_argument('--kernel', type=str, default='matern', choices=['rbf', 'matern'])
    parser.add_argument('--gp_length_scale', type=float, default=1.0)
    parser.add_argument('--gp_alpha', type=float, default=1.0)
    parser.add_argument('--hidden_units', type=int, default=100)


def get_baseline(args):

    path_args = '{}/args.pickle'.format(args.teacher_model)
    path_check = '{}/check/checkpoint.pt'.format(args.teacher_model)
    with open(path_args, 'rb') as f:
        teacher_args = pickle.load(f)

    teacher_model = UnconditionalFlow(num_flows=teacher_args.num_flows,
                                      actnorm=teacher_args.actnorm,
                                      affine=teacher_args.affine,
                                      scale_fn_str=teacher_args.scale_fn,
                                      hidden_units=teacher_args.hidden_units,
                                      activation=teacher_args.activation,
                                      range_flow=teacher_args.range_flow,
                                      augment_size=teacher_args.augment_size,
                                      base_dist=teacher_args.base_dist)

    checkpoint = torch.load(path_check)
    teacher_model.load_state_dict(checkpoint['model'])
    print('Loaded weights for teacher model at {}/{} epochs'.format(checkpoint['current_epoch'], teacher_args.epochs))


    if args.baseline == "gp":

        if args.kernel == 'matern':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(args.gp_length_scale, (1e-1, 10.0), nu=1.5)
        elif args.kernel == 'rbf':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(args.gp_length_scale, (1e-3, 1e3)) # more flexibility
        
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=args.gp_alpha)

    else:
        if args.cond_trans.lower().startswith("split") or args.cond_trans.lower().startswith("multiply"):
            cond_size = 1
        else:
            cond_size = 2

        l = 1e-4 # Lengthscale
        wr = l**2. / args.train_samples
        dr = 2. / args.train_samples
        model = DropoutNet(input_size=cond_size,
                           output_size=2,
                           hidden_units=args.hidden_units,
                           weight_regularizer=wr,
                           dropout_regularizer=dr)

    return model, teacher_model, teacher_args.dataset
