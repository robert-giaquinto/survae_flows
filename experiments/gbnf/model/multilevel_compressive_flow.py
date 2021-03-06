import torch
import torch.nn as nn

from survae.flows import CompressiveFlow
from survae.transforms import VAE, Reshape
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import AffineCouplingBijection, ScalarAffineBijection
from survae.transforms import Squeeze2d, Conv1x1, Slice, ActNormBijection2d
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal, ConditionalNormal, StandardUniform
from survae.nn.nets import ConvEncoderNet, ConvDecoderNet

from .dequantization_flow import DequantizationFlow
from .coupling import Coupling, MixtureCoupling


class MultilevelCompressiveFlow(CompressiveFlow):

    def __init__(self, data_shape, num_bits,
                 base_distribution, num_scales, num_steps, actnorm, 
                 vae_hidden_units,
                 coupling_network,
                 dequant, dequant_steps, dequant_context,
                 coupling_blocks, coupling_channels, coupling_dropout,
                 coupling_gated_conv=None, coupling_depth=None, coupling_mixtures=None):

        assert len(base_distribution) == 1, "Only a single base distribution is supported"
        transforms = []
        current_shape = data_shape
        if num_steps == 0: num_scales = 0
        
        if dequant == 'uniform' or num_steps == 0 or num_scales == 0:
            # no bijective flows defaults to only using uniform dequantization
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
                                                 dropout=coupling_dropout,
                                                 gated_conv=coupling_gated_conv,
                                                 num_mixtures=coupling_mixtures)
            transforms.append(VariationalDequantization(encoder=dequantize_flow, num_bits=num_bits))

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        for scale in range(num_scales):

            # squeeze to exchange height and width for more channels
            transforms.append(Squeeze2d())
            current_shape = (current_shape[0] * 4,
                             current_shape[1] // 2,
                             current_shape[2] // 2)

            # Dimension preserving components
            for step in range(num_steps):
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                transforms.append(Conv1x1(current_shape[0]))
                if coupling_network == "conv":
                    transforms.append(
                        Coupling(in_channels=current_shape[0],
                                 num_blocks=coupling_blocks,
                                 mid_channels=coupling_channels,
                                 depth=coupling_depth,
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
 
            # Non-dimension preserving flows: reduce the dimensionality of data by 2 (channel-wise)
            if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
            assert current_shape[0] % 2 == 0, f"Current shape {current_shape[1]}x{current_shape[2]} must be divisible by two"
            latent_size = (current_shape[0] * current_shape[1] * current_shape[2]) // 2
            
            encoder = ConditionalNormal(
                ConvEncoderNet(in_channels=current_shape[0],
                               out_channels=latent_size,
                               mid_channels=vae_hidden_units,
                               max_pool=True, batch_norm=True),
                split_dim=1)
            decoder = ConditionalNormal(
                ConvDecoderNet(in_channels=latent_size,
                               out_shape=(current_shape[0] * 2, current_shape[1], current_shape[2]),
                               mid_channels=list(reversed(vae_hidden_units)),
                               batch_norm=True,
                               in_lambda=lambda x: x.view(x.shape[0], x.shape[1], 1, 1)),
                split_dim=1)
            
            transforms.append(VAE(encoder=encoder, decoder=decoder))
            current_shape = (current_shape[0] // 2,
                             current_shape[1],
                             current_shape[2])

            if scale < num_scales - 1:
                # reshape latent sample to have height and width
                transforms.append(Reshape(input_shape=(latent_size,), output_shape=current_shape))
            
        # Base distribution for dimension preserving portion of flow
        if base_distribution == "n":
            base_dist = StandardNormal((latent_size,))
        elif base_distribution == "c":
            base_dist = ConvNormal2d((latent_size,))
        elif base_distribution == "u":
            base_dist = StandardUniform((latent_size,))
        else:
            raise ValueError("Base distribution must be one of n=Normal, u=Uniform, or c=ConvNormal")

        # for reference save the shape output by the bijective flow
        self.latent_size = latent_size
        self.flow_shape = current_shape

        super(MultilevelCompressiveFlow, self).__init__(base_dist=[None, base_dist], transforms=transforms)

