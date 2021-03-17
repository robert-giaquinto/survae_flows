import torch
import os

from model.init_model import init_model
from model.gbnf import GBNF
from model.cond_gbnf import ConditionalGBNF


def add_model_args(parser):

    # Model choice
    parser.add_argument('--flow', type=str, default='none', choices=['vae', 'mvae', 'max', 'slice', 'none'])
    parser.add_argument('--base_distribution', type=str, default='n',
                        help="String representing the base distribution(s). 'n'=Normal, 'u'=Uniform, 'c'=ConvNorm")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--super_resolution', action='store_true')
    group.add_argument('--conditional', action='store_true')

    # compressive flow parameters
    parser.add_argument('--latent_size', type=int, default=196)
    parser.add_argument('--vae_hidden_units', nargs="*", type=int, default=[])
    parser.add_argument('--vae_activation', type=str, default='relu', choices=['relu', 'none' 'elu', 'gelu', 'swish'])
    parser.add_argument('--compression_ratio', nargs="+", type=float, default=[0.5],
                        help="Percent reduction to latent space at each scale (except final). Only applies to mvae and slice flows.")

    # boosting parameters
    parser.add_argument('--boosted_components', type=int, default=1)
    parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'])
    parser.add_argument('--pretrained_model', type=str, default=None, help="Path to a pretrained flow to use as first boosting component")

    # flow parameters
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=2)

    # actnorm and if actnorm + 1x1conv should be conditional
    parser.add_argument('--actnorm', type=eval, default=True)
    parser.add_argument('--conditional_channels', nargs="*", type=int, default=[],
                        help="Number of mid channels to use for conditional actnorm and invertible 1x1 convolutional layers")

    # context init model of low-resolution images in super-image resolution
    parser.add_argument('--lowres_encoder_channels', type=int, default=0, help="Set to zero for no encoding of the low-resolution image")
    parser.add_argument('--lowres_encoder_depth', type=int, default=0, help="Set to zero for no encoding of the low-resolution image")
    parser.add_argument('--lowres_encoder_blocks', type=int, default=0, help="Set to zero for no encoding of the low-resolution image")
    parser.add_argument('--lowres_upsampler_channels', nargs="+", type=int, default=[32])

    # dequantization parameters
    parser.add_argument('--dequant', type=str, default='uniform', choices=['uniform', 'flow', 'none'])
    parser.add_argument('--dequant_steps', type=int, default=4)
    parser.add_argument('--dequant_context', type=int, default=32)

    # coupling network parameters
    parser.add_argument('--coupling_network', type=str, default="transformer", choices=["conv", "transformer", "densenet"])
    parser.add_argument('--coupling_blocks', type=int, default=1)
    parser.add_argument('--coupling_channels', type=int, default=32)
    parser.add_argument('--coupling_depth', type=int, default=1, help="Only applies to conv and densenet coupling layers")
    parser.add_argument('--coupling_dropout', type=float, default=0.0)
    parser.add_argument('--coupling_gated_conv', type=eval, default=True, help="Only applies to densenet coupling layers")
    parser.add_argument('--coupling_mixtures', type=int, default=4, help="Only applies to flow++ ('transformer') coupling layers")
    

def get_model_id(args):
    #arch = f"scales{args.num_scales}_steps{args.num_steps}_{args.coupling_network}_coupling"

    if args.flow == "vae":
        model_id = f"VAE_Compressive_Flow_latent{args.latent_size}"        
    elif args.flow == "mvae":
        model_id = 'Multilevel_Compressive_Flow'
    elif args.flow == "max":
        model_id = 'Max_Pool_Flow'
    elif args.flow == "slice":
        model_id = 'Slice_Flow'
    elif args.flow == "none":
        model_id = 'Bijective_Flow'
    else:
        raise ValueError(f"No model defined for args.flow={args.flow}")

    if args.super_resolution:
        model_id = f'Super_Resolution_{args.sr_scale_factor}x_' + model_id 
    elif args.conditional:
        model_id = 'Conditional_' + model_id

    if args.boosted_components > 1:
        model_id = f"Boosted{args.boosted_components}_" + model_id
        
    return model_id #+ "_" + arch


def get_model(args, data_shape, cond_shape=None):

    if args.boosted_components > 1:
        if args.super_resolution or args.conditional:
            model = ConditionalGBNF(data_shape=data_shape, cond_shape=cond_shape, num_bits=args.num_bits, args=args)
        else:
            model = GBNF(data_shape=data_shape, num_bits=args.num_bits, args=args)
    else:
        model = init_model(args=args, data_shape=data_shape, cond_shape=cond_shape)

    return model
