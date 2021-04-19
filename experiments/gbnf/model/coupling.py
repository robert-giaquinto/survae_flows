import torch
import torch.nn as nn
from survae.utils import sum_except_batch
from survae.transforms import AffineCouplingBijection, ConditionalAffineCouplingBijection
from survae.transforms import LogisticMixtureAffineCouplingBijection, ConditionalLogisticMixtureAffineCouplingBijection
from survae.nn.nets import DenseNet, ConvNet, ResizeConvNet, ConvDecoderNet, GatedConvNet, TransformerNet
from survae.nn.layers import ElementwiseParams2d, scale_fn


class Coupling(AffineCouplingBijection):

    def __init__(self, in_channels, num_blocks, mid_channels, depth, dropout, gated_conv, coupling_network, checkerboard=False, flip=False):

        if checkerboard:
            num_in = in_channels
            num_out = in_channels * 2
            split_dim = 3
        else:
            num_in = in_channels // 2
            num_out = 2 * (in_channels - num_in)
            split_dim = 1

        assert in_channels % 2 == 0 or split_dim != 1, f"in_channels = {in_channels} not evenly divisible"

        if coupling_network == "densenet":
            net = nn.Sequential(DenseNet(in_channels=num_in,
                                         out_channels=num_out,
                                         num_blocks=num_blocks,
                                         mid_channels=mid_channels,
                                         depth=depth,
                                         growth=mid_channels,
                                         dropout=dropout,
                                         gated_conv=gated_conv,
                                         zero_init=True),
                                ElementwiseParams2d(2, mode='sequential'))
        elif coupling_network == "conv":
            net = nn.Sequential(ConvNet(in_channels=num_in,
                                        out_channels=num_out,
                                        mid_channels=mid_channels,
                                        num_layers=depth,
                                        activation='relu'),
                                ElementwiseParams2d(2, mode='sequential'))
        else:
            raise ValueError(f"Unknown coupling_network type {coupling_network}")

        super(Coupling, self).__init__(coupling_net=net, scale_fn=scale_fn("tanh_exp"), split_dim=split_dim, flip=flip)


class MixtureCoupling(LogisticMixtureAffineCouplingBijection):

    def __init__(self, in_channels, mid_channels, num_mixtures, num_blocks, dropout, checkerboard=False, flip=False):

        if checkerboard:
            num_in = in_channels
            split_dim = 3
        else:
            num_in = in_channels // 2
            split_dim = 1

        net = nn.Sequential(TransformerNet(in_channels=num_in,
                                           mid_channels=mid_channels,
                                           num_blocks=num_blocks,
                                           num_mixtures=num_mixtures,
                                           dropout=dropout),
                                   ElementwiseParams2d(2 + num_mixtures * 3, mode='sequential'))

        super(MixtureCoupling, self).__init__(coupling_net=net, num_mixtures=num_mixtures, scale_fn=scale_fn("tanh_exp"), split_dim=split_dim, flip=flip)


class ConditionalCoupling(ConditionalAffineCouplingBijection):

    def __init__(self, in_channels, num_context, num_blocks, mid_channels, depth, dropout, gated_conv, coupling_network, checkerboard=False, flip=False):

        if checkerboard:
            num_in = in_channels + num_context
            num_out = in_channels * 2
            split_dim = 3
        else:
            num_in = in_channels // 2 + num_context
            num_out = in_channels
            split_dim = 1

        assert in_channels % 2 == 0 or split_dim != 1, f"in_channels = {in_channels} not evenly divisible"
    
        if coupling_network == "densenet":
            net = nn.Sequential(DenseNet(in_channels=num_in,
                                         out_channels=num_out,
                                         num_blocks=num_blocks,
                                         mid_channels=mid_channels,
                                         depth=depth,
                                         growth=mid_channels,
                                         dropout=dropout,
                                         gated_conv=gated_conv,
                                         zero_init=True),
                                ElementwiseParams2d(2, mode='sequential'))
        elif coupling_network == "conv":
            net = nn.Sequential(ConvNet(in_channels=num_in,
                                        out_channels=num_out,
                                        mid_channels=mid_channels,
                                        num_layers=depth,
                                        activation='relu'),
                                ElementwiseParams2d(2, mode='sequential'))
        else:
            raise ValueError(f"Unknown coupling network {coupling_network}")
            
        super(ConditionalCoupling, self).__init__(
            coupling_net=net, scale_fn=scale_fn("tanh_exp"), split_dim=split_dim, flip=flip)


class ConditionalMixtureCoupling(ConditionalLogisticMixtureAffineCouplingBijection):

    def __init__(self, in_channels, num_context, mid_channels, num_mixtures, num_blocks, dropout, use_attn=True, checkerboard=False, flip=False):

        if checkerboard:
            num_in = in_channels
            split_dim = 3
        else:
            num_in = in_channels // 2
            split_dim = 1

        coupling_net = nn.Sequential(TransformerNet(in_channels=num_in,
                                                    context_channels=num_context,
                                                    mid_channels=mid_channels,
                                                    num_blocks=num_blocks,
                                                    num_mixtures=num_mixtures,
                                                    use_attn=use_attn,
                                                    dropout=dropout),
                                     ElementwiseParams2d(2 + num_mixtures * 3, mode='sequential'))

        super(ConditionalMixtureCoupling, self).__init__(
            coupling_net=coupling_net, num_mixtures=num_mixtures, scale_fn=scale_fn("tanh_exp"), split_dim=split_dim, flip=flip)


class SRCoupling(ConditionalAffineCouplingBijection):

    def __init__(self, x_size, y_size, coupling_network, mid_channels, depth, num_blocks=None, dropout=None, gated_conv=None, checkerboard=False, flip=False):
        
        context_size = y_size
        if checkerboard:
            in_channels = y_size[0] + context_size[0]
            out_channels = y_size[0] * 2
            split_dim = 3
        else:
            in_channels = y_size[0] // 2 + context_size[0]
            out_channels = y_size[0]
            split_dim = 1

        assert x_size[1] == y_size[1] and x_size[2] == y_size[2]
        assert y_size[0] % 2 == 0 or split_dim != 1, f"High-resolution has shape {y_size} with channels not evenly divisible"

        if coupling_network == "densenet":
            coupling_net = nn.Sequential(DenseNet(in_channels=in_channels,
                                         out_channels=out_channels,
                                         num_blocks=num_blocks,
                                         mid_channels=mid_channels,
                                         depth=depth,
                                         growth=mid_channels,
                                         dropout=dropout,
                                         gated_conv=gated_conv,
                                         zero_init=True),
                                ElementwiseParams2d(2, mode='sequential'))

        elif coupling_network == "conv":
            coupling_net = nn.Sequential(ConvNet(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 mid_channels=mid_channels,
                                                 num_layers=depth,
                                                 weight_norm=True,
                                                 activation='relu'),
                                         ElementwiseParams2d(2, mode='sequential'))
            
        else:
            raise ValueError(f"Unknown coupling network {coupling_network}")
            
        super(SRCoupling, self).__init__(
            coupling_net=coupling_net, scale_fn=scale_fn("tanh_exp"), split_dim=split_dim, flip=flip)


class SRMixtureCoupling(ConditionalLogisticMixtureAffineCouplingBijection):

    def __init__(self, x_size, y_size, mid_channels, num_blocks, num_mixtures, dropout, checkerboard=False, flip=False):
        
        context_size = y_size        
        if checkerboard:
            in_channels = y_size[0]
            split_dim = 3
        else:
            in_channels = y_size[0] // 2
            split_dim = 1

        assert y_size[0] % 2 == 0 or split_dim != 1, f"High-resolution has shape {y_size} with channels not evenly divisible"
        assert x_size[1] == y_size[1] and x_size[2] == y_size[2]
        
        coupling_net = nn.Sequential(TransformerNet(in_channels=in_channels,
                                                    context_channels=context_size[0],
                                                    mid_channels=mid_channels,
                                                    num_blocks=num_blocks,
                                                    num_mixtures=num_mixtures,
                                                    dropout=dropout),
                                     ElementwiseParams2d(2 + num_mixtures * 3, mode='sequential'))
            
        super(SRMixtureCoupling, self).__init__(
            coupling_net=coupling_net, num_mixtures=num_mixtures, scale_fn=scale_fn("tanh_exp"), split_dim=split_dim, flip=flip)
