import torch
import torch.nn as nn

from survae.flows import NDPFlow
from survae.transforms import LinearVAE
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import AffineCouplingBijection, ScalarAffineBijection
from survae.transforms import Squeeze2d, Conv1x1, Slice, ActNormBijection2d
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal, ConditionalNormal

from .dequantization_flow import DequantizationFlow
from .densenet import densenet as net


class LinearFlow(NDPFlow):

    def __init__(self, data_shape, num_bits, num_scales, num_steps, actnorm, pooling,
                 dequant, dequant_steps, dequant_context,
                 densenet_blocks, densenet_channels, densenet_depth,
                 densenet_growth, dropout, gated_conv,
                 latent_size, trainable_sigma, sigma_init, stochastic_elbo):

        transforms = []
        current_shape = data_shape

        if dequant == 'uniform':
            transforms.append(UniformDequantization(num_bits=num_bits))
        elif dequant == 'flow':
            dequantize_flow = DequantizationFlow(data_shape=data_shape,
                                                 num_bits=num_bits,
                                                 num_steps=dequant_steps,
                                                 num_context=dequant_context,
                                                 num_blocks=densenet_blocks,
                                                 mid_channels=densenet_channels,
                                                 depth=densenet_depth,
                                                 growth=densenet_growth,
                                                 dropout=dropout,
                                                 gated_conv=gated_conv)
            transforms.append(VariationalDequantization(encoder=dequantize_flow, num_bits=num_bits))

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        # Initial squeeze
        transforms.append(Squeeze2d())
        current_shape = (current_shape[0] * 4,
                         current_shape[1] // 2,
                         current_shape[2] // 2)

        # Dimension preserving flows
        for scale in range(num_scales):
            for step in range(num_steps):
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                transforms.extend([
                    Conv1x1(current_shape[0]),
                    AffineCouplingBijection(
                        net(current_shape[0],
                            num_blocks=densenet_blocks,
                            mid_channels=densenet_channels,
                            depth=densenet_depth,
                            growth=densenet_growth,
                            dropout=dropout,
                            gated_conv=gated_conv,
                            zero_init=True))
                    ])

            if scale < num_scales-1:
                transforms.append(Squeeze2d())
                current_shape = (current_shape[0] * 4,
                                 current_shape[1] // 2,
                                 current_shape[2] // 2)
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))

        # Base distribution for dimension preserving portion of flow
        #base1 = ConvNormal2d(current_shape)
        base1 = StandardNormal(current_shape)

        # Non-dimension preserving flows
        input_dim = current_shape[0] * current_shape[1] * current_shape[2]
        transforms.append(LinearVAE(input_dim=input_dim,
            hidden_dim=latent_size,
            trainable_sigma=trainable_sigma,
            sigma_init=sigma_init,
            stochastic_elbo=stochastic_elbo))

        # Base distribution for non-dimension preserving portion of flow
        base2 = None  # LinearVAE class computes these terms

        super(LinearFlow, self).__init__(base_dist=[base1, base2], transforms=transforms)



