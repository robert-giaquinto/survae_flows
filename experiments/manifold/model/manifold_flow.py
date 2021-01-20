import torch
import torch.nn as nn

from survae.flows import NDPFlow
from survae.transforms import VAE
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import AffineCouplingBijection, ScalarAffineBijection
from survae.transforms import Squeeze2d, Conv1x1, Slice, ActNormBijection2d
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal, ConditionalNormal, StandardUniform
from survae.nn.nets import MLP

from .dequantization_flow import DequantizationFlow
from .coupling import Coupling, MixtureCoupling


class ManifoldFlow(NDPFlow):

    def __init__(self, data_shape, num_bits,
                 base_distributions, num_scales, num_steps, actnorm, 
                 vae_hidden_units, latent_size, vae_activation,                 
                 coupling_network,
                 dequant, dequant_steps, dequant_context,
                 coupling_blocks, coupling_channels, coupling_dropout,
                 coupling_growth=None, coupling_gated_conv=None, coupling_depth=None, coupling_mixtures=None):

        transforms = []
        current_shape = data_shape

        if dequant == 'uniform':
            transforms.append(UniformDequantization(num_bits=num_bits))
        elif dequant == 'flow':            
            dequantize_flow = DequantizationFlow(data_shape=data_shape,
                                                 num_bits=num_bits,
                                                 num_steps=dequant_steps,
                                                 coupling_network=coupling_network,
                                                 num_context=dequant_context,
                                                 num_blocks=coupling_blocks,
                                                 mid_channels=coupling_channels,
                                                 depth=coupling_depth,
                                                 growth=coupling_growth,
                                                 dropout=coupling_dropout,
                                                 gated_conv=coupling_gated_conv,
                                                 num_mixtures=coupling_mixtures)
            transforms.append(VariationalDequantization(encoder=dequantize_flow, num_bits=num_bits))

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        # Initial squeeze
        transforms.append(Squeeze2d())
        current_shape = (current_shape[0] * 4,
                         current_shape[1] // 2,
                         current_shape[2] // 2)

        
        # Dimension preserving flows
        if num_steps == 0: num_scales = 0
        for scale in range(num_scales):
            for step in range(num_steps):
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                transforms.append(Conv1x1(current_shape[0]))
                if coupling_network in ["conv", "densenet"]:
                    transforms.append(
                        Coupling(in_channels=current_shape[0],
                                 num_blocks=coupling_blocks,
                                 mid_channels=coupling_channels,
                                 depth=coupling_depth,
                                 growth=coupling_growth,
                                 dropout=coupling_dropout,
                                 gated_conv=coupling_gated_conv,
                                 coupling_network=coupling_network))
                else:
                    transforms.append(
                        MixtureCoupling(in_channels=current_shape[0],
                                        mid_channels=coupling_channels,
                                        num_mixtures=coupling_mixtures,
                                        num_blocks=coupling_blocks,
                                        dropout=coupling_dropout))

            if scale < num_scales-1:
                transforms.append(Squeeze2d())
                current_shape = (current_shape[0] * 4,
                                 current_shape[1] // 2,
                                 current_shape[2] // 2)
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))

        # Base distribution for dimension preserving portion of flow
        if len(base_distributions) > 1:
            if base_distributions[0] == "n":
                base0 = StandardNormal(current_shape)
            elif base_distributions[0] == "c":
                base0 = ConvNormal2d(current_shape)
            elif base_distributions[0] == "u":
                base0 = StandardUniform(current_shape)
            else:
                raise ValueError("Base distribution must be one of n=Noraml, u=Uniform, or c=ConvNormal")
        else:
            base0 = None

        # for reference save the shape output by the bijective flow
        self.flow_shape = current_shape

        # Non-dimension preserving flows
        flat_dim = current_shape[0] * current_shape[1] * current_shape[2]
        # conv_vae = True
        # if conv_vae:
        #     encoder = ConditionalNormal(
        #         GatedConv2dNet(in_channels=current_shape[0],
        #                     out_channels=2 * latent_size,
        #                     hidden_channels=64,
        #                     activation="none",
        #                     out_lambda=lambda x: x.view(x.shape[0], flat_dim)))
        #     decoder = ConditionalNormal(
        #         GatedConvTranspose2dNet(in_channels=latent_size,
        #                              out_channels=2 * flat_dim    or current_shape[0],
        #                              hidden_channels=64,
        #                              activation="none",
        #                              in_lambda=lambda x: x.view(x.shape[0], latent_size, 1, 1)),
        #         split_dim=1)
            
        # else:
        encoder = ConditionalNormal(MLP(flat_dim, 2 * latent_size,
                                        hidden_units=vae_hidden_units,
                                        activation=vae_activation,
                                        in_lambda=lambda x: x.view(x.shape[0], flat_dim)))
        decoder = ConditionalNormal(MLP(latent_size, 2 * flat_dim,
                                        hidden_units=list(reversed(vae_hidden_units)),
                                        activation=vae_activation,
                                        out_lambda=lambda x: x.view(x.shape[0], current_shape[0]*2, current_shape[1], current_shape[2])), split_dim=1)
        
        transforms.append(VAE(encoder=encoder, decoder=decoder))

        # Base distribution for non-dimension preserving portion of flow
        #self.latent_size = latent_size
        if base_distributions[-1] == "n":
            base1 = StandardNormal((latent_size,))
        elif base_distributions[-1] == "c":
            base1 = ConvNormal2d((latent_size,))
        elif base_distributions[-1] == "u":
            base1 = StandardUniform((latent_size,))
        else:
            raise ValueError("Base distribution must be one of n=Noraml, u=Uniform, or c=ConvNormal")

        super(ManifoldFlow, self).__init__(base_dist=[base0, base1], transforms=transforms)

