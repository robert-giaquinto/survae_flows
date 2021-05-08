import torch
import torch.nn as nn

from survae.flows import Flow
from survae.transforms import UniformDequantization, VariationalDequantization, Augment
from survae.transforms import AffineCouplingBijection, ScalarAffineBijection, SimpleMaxPoolSurjection2d
from survae.transforms import Squeeze2d, Conv1x1, Slice, ActNormBijection2d
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal

from model.coupling import Coupling, MixtureCoupling
from model.dequantization_flow import DequantizationFlow
from model.augment_flow import AugmentFlow



class PoolFlow(Flow):

    def __init__(self, data_shape, num_bits,
                 num_scales, num_steps, actnorm,
                 pooling, compression_ratio,
                 coupling_network,
                 coupling_blocks, coupling_channels,
                 coupling_dropout=0.0, coupling_gated_conv=None, coupling_depth=None, coupling_mixtures=None,
                 dequant="flow", dequant_steps=4, dequant_context=32, dequant_blocks=2,
                 augment_steps=4, augment_context=32, augment_blocks=2, augment_size=None,
                 checkerboard_scales=[], tuple_flip=True):


        if len(compression_ratio) == 1 and num_scales > 1:
            compression_ratio = [compression_ratio[0]] * (num_scales - 1)
        assert all([compression_ratio[s] >= 0 and compression_ratio[s] < 1 for s in range(num_scales-1)])

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
                                                 num_blocks=dequant_blocks,
                                                 mid_channels=coupling_channels,
                                                 depth=coupling_depth,
                                                 dropout=0.0,
                                                 gated_conv=False,
                                                 num_mixtures=coupling_mixtures,
                                                 checkerboard=True,
                                                 tuple_flip=tuple_flip)
            transforms.append(VariationalDequantization(encoder=dequantize_flow, num_bits=num_bits))

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        # Initial squeezing
        if current_shape[1] >= 128 and current_shape[2] >= 128:
            # H x W -> 64 x 64
            transforms.append(Squeeze2d())
            current_shape = (current_shape[0] * 4, current_shape[1] // 2, current_shape[2] // 2)

        if current_shape[1] >= 64 and current_shape[2] >= 64:
            # H x W -> 32 x 32
            transforms.append(Squeeze2d())
            current_shape = (current_shape[0] * 4, current_shape[1] // 2, current_shape[2] // 2)

        if 0 not in checkerboard_scales or (current_shape[1] > 32 and current_shape[2] > 32):
            # Only go to 16 x 16 if not doing checkerboard splits first
            transforms.append(Squeeze2d())
            current_shape = (current_shape[0] * 4, current_shape[1] // 2, current_shape[2] // 2)

        # add in augmentation channels if desired
        if augment_size is not None and augment_size > 0:
            augment_flow = AugmentFlow(data_shape=current_shape,
                                       augment_size=augment_size,
                                       num_steps=augment_steps,
                                       coupling_network=coupling_network,
                                       mid_channels=coupling_channels,
                                       num_context=augment_context,
                                       num_mixtures=coupling_mixtures,
                                       num_blocks=augment_blocks,
                                       dropout=0.0,
                                       checkerboard=True,
                                       tuple_flip=tuple_flip)
            transforms.append(Augment(encoder=augment_flow, x_size=current_shape[0]))
            current_shape = (current_shape[0] + augment_size,
                             current_shape[1],
                             current_shape[2])

        for scale in range(num_scales):
            checkerboard = scale in checkerboard_scales
            
            for step in range(num_steps):
                flip = (step % 2 == 0) if tuple_flip else False

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
                                 coupling_network=coupling_network,
                                 checkerboard=checkerboard,
                                 flip=flip))
                else:
                    transforms.append(
                        MixtureCoupling(in_channels=current_shape[0],
                                        mid_channels=coupling_channels,
                                        num_mixtures=coupling_mixtures,
                                        num_blocks=coupling_blocks,
                                        dropout=coupling_dropout,
                                        checkerboard=checkerboard,
                                        flip=flip))

            if scale < num_scales-1:
                if pooling in ['bijective', 'none'] or compression_ratio[scale] == 0.0:
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
