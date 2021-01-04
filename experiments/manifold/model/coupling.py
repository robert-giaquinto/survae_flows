import torch
import torch.nn as nn
from survae.utils import sum_except_batch
from survae.transforms import AffineCouplingBijection, ConditionalAffineCouplingBijection
from survae.transforms import LogisticMixtureAffineCouplingBijection, ConditionalLogisticMixtureAffineCouplingBijection
from survae.nn.nets import DenseNet, ConvNet, GatedConvNet, TransformerNet
from survae.nn.layers import ElementwiseParams2d, scale_fn


class Coupling(AffineCouplingBijection):

    def __init__(self, in_channels, num_blocks, mid_channels, depth, growth, dropout, gated_conv, coupling_network):

        assert in_channels % 2 == 0

        coupling_in_size = in_channels // 2
        coupling_out_size = 2 * (in_channels - coupling_in_size)
        if coupling_network == "densenet":
            net = nn.Sequential(DenseNet(in_channels=coupling_in_size,
                                         out_channels=coupling_out_size,
                                         num_blocks=num_blocks,
                                         mid_channels=mid_channels,
                                         depth=depth,
                                         growth=growth,
                                         dropout=dropout,
                                         gated_conv=gated_conv,
                                         zero_init=True),
                                ElementwiseParams2d(2, mode='sequential'))
        elif coupling_network == "conv":
            net = nn.Sequential(ConvNet(in_channels=coupling_in_size,
                                        out_channels=coupling_out_size,
                                        mid_channels=mid_channels,
                                        num_layers=depth,
                                        activation='relu'),
                                ElementwiseParams2d(2, mode='sequential'))
        else:
            raise ValueError(f"Unknown coupling_network type {coupling_network}")

        super(Coupling, self).__init__(coupling_net=net, scale_fn=scale_fn("tanh_exp"))

    # def _elementwise_forward(self, x, elementwise_params):
    #     unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
    #     log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
    #     z = shift + torch.exp(log_scale) * x
    #     ldj = sum_except_batch(log_scale)
    #     return z, ldj

    # def _elementwise_inverse(self, z, elementwise_params):
    #     unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
    #     log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
    #     x = (z - shift) * torch.exp(-log_scale)
    #     return x


class MixtureCoupling(LogisticMixtureAffineCouplingBijection):

    def __init__(self, in_channels, mid_channels, num_mixtures, num_blocks, dropout):

        net = nn.Sequential(TransformerNet(in_channels // 2,
                                           mid_channels=mid_channels,
                                           num_blocks=num_blocks,
                                           num_mixtures=num_mixtures,
                                           dropout=dropout),
                                   ElementwiseParams2d(2 + num_mixtures * 3, mode='sequential'))

        super(MixtureCoupling, self).__init__(coupling_net=net, num_mixtures=num_mixtures, scale_fn=scale_fn("tanh_exp"))




class ConditionalCoupling(ConditionalAffineCouplingBijection):

    def __init__(self, in_channels, num_context, num_blocks, mid_channels, depth, growth, dropout, gated_conv, coupling_network):

        assert in_channels % 2 == 0

        if coupling_network == "densenet":
            net = nn.Sequential(DenseNet(in_channels=in_channels // 2 + num_context,
                                         out_channels=in_channels,
                                         num_blocks=num_blocks,
                                         mid_channels=mid_channels,
                                         depth=depth,
                                         growth=growth,
                                         dropout=dropout,
                                         gated_conv=gated_conv,
                                         zero_init=True),
                                ElementwiseParams2d(2, mode='sequential'))
        elif coupling_network == "conv":
            net = nn.Sequential(ConvNet(in_channels=in_channels // 2 + num_context,
                                        out_channels=in_channels,
                                        mid_channels=mid_channels,
                                        num_layers=depth,
                                        activation='relu'),
                                ElementwiseParams2d(2, mode='sequential'))
        else:
            raise ValueError(f"Unknown coupling network {coupling_network}")
            
        super(ConditionalCoupling, self).__init__(coupling_net=net, scale_fn=scale_fn("tanh_exp"))

    # def _elementwise_forward(self, x, elementwise_params):
    #     unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
    #     log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
    #     z = shift + torch.exp(log_scale) * x
    #     ldj = sum_except_batch(log_scale)
    #     return z, ldj

    # def _elementwise_inverse(self, z, elementwise_params):
    #     unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
    #     log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
    #     x = (z - shift) * torch.exp(-log_scale)
    #     return x


class ConditionalMixtureCoupling(ConditionalLogisticMixtureAffineCouplingBijection):

    def __init__(self, in_channels, num_context, mid_channels, num_mixtures, num_blocks, dropout):

        net = nn.Sequential(TransformerNet(in_channels // 2,
                                           mid_channels=mid_channels,
                                           context_channels=num_context,
                                           num_blocks=num_blocks,
                                           num_mixtures=num_mixtures,
                                           dropout=dropout),
                            ElementwiseParams2d(2 + num_mixtures * 3, mode='sequential'))

        super(ConditionalMixtureCoupling, self).__init__(coupling_net=net, num_mixtures=num_mixtures, scale_fn=scale_fn("tanh_exp"))
