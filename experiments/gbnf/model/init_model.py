import torch
import os

from model.manifold_flow import ManifoldFlow
from model.multilevel_flow import MultilevelFlow
from model.pool_flow import PoolFlow
from model.cond_pool_flow import CondPoolFlow
from model.sr_pool_flow import SRPoolFlow


def init_model(args, data_shape, cond_shape=None):
    """
    Putting this function in a seperate file to avoid a circular reference, since
    GBNF uses the same function to build the ensemble
    """
    if args.flow == "sr":
        model = SRPoolFlow(data_shape=data_shape,
                           cond_shape=cond_shape,
                           num_bits=args.num_bits,
                           num_scales=args.num_scales,
                           num_steps=args.num_steps,
                           actnorm=args.actnorm,
                           pooling="none",
                           dequant=args.dequant,
                           dequant_steps=args.dequant_steps,
                           dequant_context=args.dequant_context,
                           coupling_network=args.coupling_network,
                           coupling_blocks=args.coupling_blocks,
                           coupling_channels=args.coupling_channels,
                           coupling_depth=args.coupling_depth,
                           coupling_dropout=args.coupling_dropout,
                           coupling_gated_conv=args.coupling_gated_conv,
                           coupling_mixtures=args.coupling_mixtures)
        
    elif args.flow == "vae":
        model = ManifoldFlow(data_shape=data_shape,
                             num_bits=args.num_bits,
                             base_distribution=args.base_distribution,
                             num_scales=args.num_scales,
                             num_steps=args.num_steps,
                             actnorm=args.actnorm,
                             latent_size=args.latent_size,
                             vae_hidden_units=args.vae_hidden_units,
                             vae_activation=args.vae_activation,
                             dequant=args.dequant,
                             dequant_steps=args.dequant_steps,
                             dequant_context=args.dequant_context,
                             coupling_network=args.coupling_network,
                             coupling_blocks=args.coupling_blocks,
                             coupling_channels=args.coupling_channels,
                             coupling_depth=args.coupling_depth,
                             coupling_dropout=args.coupling_dropout,
                             coupling_gated_conv=args.coupling_gated_conv,
                             coupling_mixtures=args.coupling_mixtures)
        
    elif args.flow == "mvae":
        model = MultilevelFlow(data_shape=data_shape,
                               num_bits=args.num_bits,
                               base_distribution=args.base_distribution,
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
                               coupling_dropout=args.coupling_dropout,
                               coupling_gated_conv=args.coupling_gated_conv,
                               coupling_mixtures=args.coupling_mixtures)

            
    elif args.flow in ["max", "slice", "none"]:
        
        model = PoolFlow(data_shape=data_shape,
                         num_bits=args.num_bits,
                         num_scales=args.num_scales,
                         num_steps=args.num_steps,
                         actnorm=args.actnorm,
                         pooling=args.flow,
                         dequant=args.dequant,
                         dequant_steps=args.dequant_steps,
                         dequant_context=args.dequant_context,
                         coupling_network=args.coupling_network,
                         coupling_blocks=args.coupling_blocks,
                         coupling_channels=args.coupling_channels,
                         coupling_depth=args.coupling_depth,
                         coupling_dropout=args.coupling_dropout,
                         coupling_gated_conv=args.coupling_gated_conv,
                         coupling_mixtures=args.coupling_mixtures)

    else:
        raise ValueError(f"No model defined for {args.flow} flows")

    return model
