import torch
import os

from model.linear_manifold_flow import LinearManifoldFlow
from model.manifold_flow import ManifoldFlow
from model.multilevel_flow import MultilevelFlow
from model.pool_flow import PoolFlow
from model.pretrained_flow import PretrainedFlow
from model.compress_pretrained import CompressPretrained


def add_model_args(parser):

    # Model choice
    parser.add_argument('--compression', type=str, default='none', choices={'vae', 'mvae', 'max', 'slice', 'none', 'pretrained'})
    parser.add_argument('--base_distributions', type=str, default='n',
                        help="String representing the base distribution(s). 'n'=Normal, 'u'=Uniform, 'c'=ConvNorm")
    parser.add_argument('--latent_size', type=int, default=196)
    parser.add_argument('--vae_hidden_units', nargs="*", type=int, default=[512, 256])
    parser.add_argument('--vae_activation', type=str, default='none')

    # Linear Manifold Flow params
    parser.add_argument('--linear', type=eval, default=True)
    parser.add_argument('--stochastic_elbo', type=eval, default=False)
    parser.add_argument('--trainable_sigma', type=eval, default=True)
    parser.add_argument('--sigma_init', type=float, default=1.0)

    # Flow params
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--actnorm', type=eval, default=True)

    # Dequant params
    parser.add_argument('--dequant', type=str, default='uniform', choices={'uniform', 'flow', 'none'})
    parser.add_argument('--dequant_steps', type=int, default=4)
    parser.add_argument('--dequant_context', type=int, default=32)

    # Net params
    parser.add_argument('--coupling_network', type=str, default="transformer", choices=["densenet", "conv", "transformer"])
    parser.add_argument('--coupling_blocks', type=int, default=1)
    parser.add_argument('--coupling_channels', type=int, default=64)
    parser.add_argument('--coupling_depth', type=int, default=1)
    parser.add_argument('--coupling_growth', type=int, default=16)
    parser.add_argument('--coupling_dropout', type=float, default=0.2)
    parser.add_argument('--coupling_gated_conv', type=eval, default=True)
    parser.add_argument('--coupling_mixtures', type=int, default=16)
    

def get_model_id(args):
    if args.compression == 'pretrained':
        model_id = f"Compress_Pretrained_{args.project}_nonpool_VAE_{'_'.join([str(elt) for elt in args.vae_hidden_units])}_Flow_latent{args.latent_size}"
        return model_id

    else:        
        arch = f"scales{args.num_scales}_steps{args.num_steps}_{args.coupling_network}"
        if args.compression == "vae":

            if args.linear:
                if args.stochastic_elbo:
                    model_id = f'NDP_Linear_Stochastic_Flow_latent{args.latent_size}_base{args.base_distributions}'
                else:
                    model_id = f'NDP_Linear_Analytical_Flow_latent{args.latent_size}_base{args.base_distributions}'
            else:
                model_id = f"NDP_VAE_{'_'.join([str(elt) for elt in args.vae_hidden_units])}_Flow_latent{args.latent_size}_base{args.base_distributions}"

        elif args.compression == "mvae":
            model_id = f"Multilevel_Flow_base{args.base_distributions}"

        elif args.compression == "max":
            model_id = 'Max_Pool_Flow'
        elif args.compression == "slice":
            model_id = 'Slice_Pool_Flow'
        elif args.compression == "none":
            model_id = 'Bijective_Flow'
        else:
            raise ValueError(f"No model defined for {args.compression} forms of dimension changes")

        return model_id + "_" + arch


def get_model(args, data_shape, cond_shape=None):

    if args.compression == 'pretrained':
        pretrained_model = PretrainedFlow(data_shape=data_shape,
                                          num_bits=args.num_bits,
                                          num_scales=args.num_scales,
                                          num_steps=args.num_steps,
                                          actnorm=args.actnorm,
                                          pooling=args.pooling,
                                          dequant=args.dequant,
                                          dequant_steps=args.dequant_steps,
                                          dequant_context=args.dequant_context,
                                          densenet_blocks=args.densenet_blocks,
                                          densenet_channels=args.densenet_channels,
                                          densenet_depth=args.densenet_depth,
                                          densenet_growth=args.densenet_growth,
                                          dropout=args.dropout,
                                          gated_conv=args.gated_conv)

        
        if hasattr(args, 'load_pretrained_weights') and args.load_pretrained_weights:
            # Load the pretrained weights from the file
            # we don't need to load the weights when fine-tuning the whole model (which happens when calling train_more.py)
            
            # Load checkpoint
            assert hasattr(args, 'pretrained_model')
            check_path = f'{args.pretrained_model}/check/'
            #if args.new_device is not None:
            checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pt'), map_location=torch.device(args.new_device))
            #else:
            #    checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pt'))

            pretrained_model.load_state_dict(checkpoint['model'])

            # freeze the pretrained model's layers
            if args.freeze:
                for param in pretrained_model.parameters():
                    param.requires_grad = False
                
            # modify model
            #if args.new_device is not None:
            pretrained_model.to(torch.device(args.device))
        
        model = CompressPretrained(pretrained_model=pretrained_model, latent_size=args.latent_size)
    
    elif args.compression == "vae":
        
        if args.linear:
            model = LinearManifoldFlow(data_shape=data_shape,
                                       num_bits=args.num_bits,
                                       base_distributions=args.base_distributions,
                                       num_scales=args.num_scales,
                                       num_steps=args.num_steps,
                                       actnorm=args.actnorm,
                                       latent_size=args.latent_size,
                                       trainable_sigma=args.trainable_sigma,
                                       sigma_init=args.sigma_init,
                                       stochastic_elbo=args.stochastic_elbo,
                                       dequant=args.dequant,
                                       dequant_steps=args.dequant_steps,
                                       dequant_context=args.dequant_context,
                                       coupling_network=args.coupling_network,
                                       coupling_blocks=args.coupling_blocks,
                                       coupling_channels=args.coupling_channels,
                                       coupling_depth=args.coupling_depth,
                                       coupling_growth=args.coupling_growth,
                                       coupling_dropout=args.coupling_dropout,
                                       coupling_gated_conv=args.coupling_gated_conv,
                                       coupling_mixtures=args.coupling_mixtures)            
        else:
            model = ManifoldFlow(data_shape=data_shape,
                                 num_bits=args.num_bits,
                                 base_distributions=args.base_distributions,
                                 num_scales=args.num_scales,
                                 num_steps=args.num_steps,
                                 actnorm=args.actnorm,
                                 vae_hidden_units=args.vae_hidden_units,
                                 latent_size=args.latent_size,
                                 vae_activation=args.vae_activation,
                                 dequant=args.dequant,
                                 dequant_steps=args.dequant_steps,
                                 dequant_context=args.dequant_context,
                                 coupling_network=args.coupling_network,
                                 coupling_blocks=args.coupling_blocks,
                                 coupling_channels=args.coupling_channels,
                                 coupling_depth=args.coupling_depth,
                                 coupling_growth=args.coupling_growth,
                                 coupling_dropout=args.coupling_dropout,
                                 coupling_gated_conv=args.coupling_gated_conv,
                                 coupling_mixtures=args.coupling_mixtures)
            
    elif args.compression == "mvae":
        model = MultilevelFlow(data_shape=data_shape,
                               num_bits=args.num_bits,
                               base_distributions=args.base_distributions,
                               num_scales=args.num_scales,
                               num_steps=args.num_steps,
                               actnorm=args.actnorm,
                               vae_hidden_units=args.vae_hidden_units,
                               vae_activation=args.vae_activation,
                               dequant=args.dequant,
                               dequant_steps=args.dequant_steps,
                               dequant_context=args.dequant_context,
                               coupling_network=args.coupling_network,
                               coupling_blocks=args.coupling_blocks,
                               coupling_channels=args.coupling_channels,
                               coupling_depth=args.coupling_depth,
                               coupling_growth=args.coupling_growth,
                               coupling_dropout=args.coupling_dropout,
                               coupling_gated_conv=args.coupling_gated_conv,
                               coupling_mixtures=args.coupling_mixtures)

            
    elif args.compression in ["max", "slice", "none"]:
        
        # Pooling surjective flow
        model = PoolFlow(data_shape=data_shape,
                         num_bits=args.num_bits,
                         num_scales=args.num_scales,
                         num_steps=args.num_steps,
                         actnorm=args.actnorm,
                         pooling=args.compression,
                         dequant=args.dequant,
                         dequant_steps=args.dequant_steps,
                         dequant_context=args.dequant_context,
                         coupling_network=args.coupling_network,
                         coupling_blocks=args.coupling_blocks,
                         coupling_channels=args.coupling_channels,
                         coupling_depth=args.coupling_depth,
                         coupling_growth=args.coupling_growth,
                         coupling_dropout=args.coupling_dropout,
                         coupling_gated_conv=args.coupling_gated_conv,
                         coupling_mixtures=args.coupling_mixtures)

    else:
        raise ValueError(f"No model defined for {args.compresion} compressions")

    return model
