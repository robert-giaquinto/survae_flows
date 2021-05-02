import torch
import os
import pickle
import argparse

from model.sr_flow import SRFlow
from model.unconditional_flow import UnconditionalFlow


def add_model_args(parser):

    # Model params
    parser.add_argument('--num_flows', type=int, default=4)
    parser.add_argument('--actnorm', type=eval, default=False)
    parser.add_argument('--affine', type=eval, default=True)
    parser.add_argument('--scale_fn', type=str, default='exp', choices={'exp', 'softplus', 'sigmoid', 'tanh_exp'})
    parser.add_argument('--hidden_units', type=eval, default=[50], nargs="+")
    parser.add_argument('--activation', type=str, default='relu', choices={'relu', 'elu', 'gelu'})
    parser.add_argument('--range_flow', type=str, default='logit', choices={'logit', 'softplus'})
    parser.add_argument('--augment_size', type=int, default=0)
    parser.add_argument('--base_dist', type=str, default='uniform', choices={'uniform', 'normal'})


def get_model_id(args):

    if args.teacher_model is not None:
        model_id = f"Student"
    else:
        model_id = f"Teacher"

    return model_id


def get_model(args):

    if args.teacher_model is None:
        model = UnconditionalFlow(num_flows=args.num_flows,
                                  actnorm=args.actnorm,
                                  affine=args.affine,
                                  scale_fn_str=args.scale_fn,
                                  hidden_units=args.hidden_units,
                                  activation=args.activation,
                                  range_flow=args.range_flow,
                                  augment_size=args.augment_size,
                                  base_dist=args.base_dist)
        return model
    
    else:
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

        if args.cond_trans.lower().startswith("split") or args.cond_trans.lower().startswith("multiply"):
            cond_size = 1
        else:
            cond_size = 2
        
        student_model = SRFlow(num_flows=args.num_flows,
                               actnorm=args.actnorm,
                               affine=args.affine,
                               scale_fn_str=args.scale_fn,
                               hidden_units=args.hidden_units,
                               activation=args.activation,
                               range_flow=args.range_flow,
                               augment_size=args.augment_size,
                               base_dist=args.base_dist,
                               cond_size=cond_size)
        
        return student_model, teacher_model, teacher_args.dataset
