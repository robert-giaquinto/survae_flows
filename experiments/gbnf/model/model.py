import torch
import os

from model.init_model import init_model
from model.manifold_flow import ManifoldFlow
from model.multilevel_flow import MultilevelFlow
from model.pool_flow import PoolFlow
from model.unconditional_gbnf import GBNF


def add_model_args(parser):

    # Model choice
    parser.add_argument('--flow', type=str, default='none', choices=['vae', 'mvae', 'max', 'slice', 'none', 'sr', 'conditional'])
    parser.add_argument('--base_distribution', type=str, default='n',
                        help="String representing the base distribution(s). 'n'=Normal, 'u'=Uniform, 'c'=ConvNorm")

    # compressive flow parameters
    parser.add_argument('--latent_size', type=int, default=196)
    parser.add_argument('--vae_hidden_units', nargs="*", type=int, default=[])
    parser.add_argument('--vae_activation', type=str, default='relu', choices=['relu', 'none' 'elu', 'gelu', 'swish'])

    # boosting parameters
    parser.add_argument('--boosted_components', type=int, default=1)
    parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'])

    # flow parameters
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--actnorm', type=eval, default=True)

    # dequantization parameters
    parser.add_argument('--dequant', type=str, default='uniform', choices=['uniform', 'flow', 'none'])
    parser.add_argument('--dequant_steps', type=int, default=4)
    parser.add_argument('--dequant_context', type=int, default=32)

    # coupling network parameters
    parser.add_argument('--coupling_network', type=str, default="transformer", choices=["conv", "transformer"])
    parser.add_argument('--coupling_blocks', type=int, default=1)
    parser.add_argument('--coupling_channels', type=int, default=64)
    parser.add_argument('--coupling_depth', type=int, default=1)
    parser.add_argument('--coupling_dropout', type=float, default=0.2)
    parser.add_argument('--coupling_gated_conv', type=eval, default=True)
    parser.add_argument('--coupling_mixtures', type=int, default=16)
    

def get_model_id(args):
    arch = f"scales{args.num_scales}_steps{args.num_steps}_{args.coupling_network}"
    if args.flow == "vae":
        model_id = f"NDP_VAE_{'_'.join([str(elt) for elt in args.vae_hidden_units])}_Flow_latent{args.latent_size}_base{args.base_distribution}"        

    elif args.flow == "mvae":
        model_id = f"Multilevel_Flow_base{args.base_distribution}"
        
    elif args.flow == "max":
        model_id = 'Max_Pool_Flow'
    elif args.flow == "slice":
        model_id = 'Slice_Flow'
    elif args.flow == "none":
        model_id = 'Bijective_Flow'
    elif args.flow == 'sr':
        model_id = 'Super_Resolution_Bijective_Flow'
    elif args.flow == 'conditional':
        model_id = 'Conditional_Bijective_Flow'
    else:
        raise ValueError(f"No model defined for args.flow={args.flow}")

    if args.boosted_components > 1:
        model_id = f"Boosted{args.boosted_components}_" + model_id
        
    return model_id + "_" + arch


def get_model(args, data_shape, cond_shape=None):

    if args.boosted_components > 1:
        model = GBNF(data_shape=data_shape, num_bits=args.num_bits, args=args)
    else:
        model = init_model(args, data_shape, cond_shape)

    return model
