import torch
import torch.nn as nn
from survae.flows import Flow
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import AffineCouplingBijection, ScalarAffineBijection, SimpleMaxPoolSurjection2d
from survae.transforms import Squeeze2d, Conv1x1, Slice, ActNormBijection2d
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal

from .coupling import Coupling, MixtureCoupling
from .dequantization_flow import DequantizationFlow


class PoolFlow(Flow):

    def __init__(self, data_shape, num_bits,
                 num_scales, num_steps, actnorm, pooling, sliced_ratio,
                 coupling_network,
                 dequant, dequant_steps, dequant_context,
                 coupling_blocks, coupling_channels, coupling_dropout,
                 coupling_gated_conv=None, coupling_depth=None, coupling_mixtures=None):

        if len(sliced_ratio) == 1 and num_scales > 1:
            sliced_ratio = [sliced_ratio[0]] * num_scales
        assert all([sliced_ratio[s] > 0 and sliced_ratio[s] <= 1 for s in range(num_scales)])
        
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
                                                 num_blocks=4,
                                                 mid_channels=coupling_channels,
                                                 depth=3,
                                                 dropout=0.0,
                                                 gated_conv=False,
                                                 num_mixtures=4)
            transforms.append(VariationalDequantization(encoder=dequantize_flow, num_bits=num_bits))

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        # Initial squeeze
        transforms.append(Squeeze2d())
        current_shape = (current_shape[0] * 4,
                         current_shape[1] // 2,
                         current_shape[2] // 2)

        # Pooling flows
        for scale in range(num_scales):
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

            if scale < num_scales-1:
                if pooling == 'none':
                    transforms.append(Squeeze2d())
                    current_shape = (current_shape[0] * 4,
                                     current_shape[1] // 2,
                                     current_shape[2] // 2)
                elif pooling == 'slice':
                    # slice some of the dimensions (channel-wise) out from further flow steps
                    unsliced_channels = int(max(1, 4  * current_shape[0] * (1.0 - sliced_ratio[scale])))
                    sliced_channels = int(4 * current_shape[0] - unsliced_channels)
                    noise_shape = (sliced_channels,
                                   current_shape[1] // 2,
                                   current_shape[2] // 2)
                    transforms.append(Squeeze2d())
                    transforms.append(Slice(StandardNormal(noise_shape), num_keep=unsliced_channels, dim=1))
                    current_shape = (unsliced_channels,
                                     current_shape[1] // 2,
                                     current_shape[2] // 2)
                elif pooling == 'max':
                    noise_shape = (current_shape[0] * 3,
                                   current_shape[1] // 2,
                                   current_shape[2] // 2)
                    decoder = StandardHalfNormal(noise_shape)
                    transforms.append(SimpleMaxPoolSurjection2d(decoder=decoder))
                    current_shape = (current_shape[0],
                                     current_shape[1] // 2,
                                     current_shape[2] // 2)

                else:
                    raise ValueError(f"Pooling argument must be either slice, max or none, not: {pooling}")
                
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))


        # for reference save the shape output by the bijective flow
        self.flow_shape = current_shape
        self.latent_size = current_shape[0] * current_shape[1] * current_shape[2]

        super(PoolFlow, self).__init__(base_dist=ConvNormal2d(current_shape), transforms=transforms)
