import torch
import torch.nn as nn

from survae.flows import ConditionalInverseFlow
from survae.distributions import ConvNormal2d, StandardNormal
from survae.transforms import Unsqueeze2d, Sigmoid, Conv1x1, ActNormBijection2d
from survae.nn.layers import LambdaLayer, Conv2d, Conv2dZeros, Checkerboard
from survae.nn.blocks import DenseBlock
from survae.nn.nets import ConvNet, GatedConvNet, TransformerNet, ConvAttnBlock

from .coupling import ConditionalCoupling, ConditionalMixtureCoupling


class AugmentFlow(ConditionalInverseFlow):

    def __init__(self, data_shape, augment_size, num_steps,
                 mid_channels, num_context, num_blocks, dropout, num_mixtures, checkerboard=True, tuple_flip=True, coupling_network="transformer"):

        context_in = data_shape[0] * 2 if checkerboard else data_shape[0]
        layers = []
        if checkerboard: layers.append(Checkerboard(concat_dim=1))
        layers += [Conv2d(context_in, mid_channels // 2, kernel_size=3, stride=1),
                   nn.Conv2d(mid_channels // 2, mid_channels, kernel_size=2, stride=2, padding=0),
                   GatedConvNet(channels=mid_channels, num_blocks=2, dropout=0.0),
                   Conv2dZeros(in_channels=mid_channels, out_channels=num_context)]
        context_net = nn.Sequential(*layers)
        
        # layer transformations of the augment flow
        transforms = []
        sample_shape = (augment_size * 4, data_shape[1] // 2, data_shape[2] // 2)
        for i in range(num_steps):
            flip = (i % 2 == 0) if tuple_flip else False
            transforms.append(ActNormBijection2d(sample_shape[0]))
            transforms.extend([Conv1x1(sample_shape[0])])
            if coupling_network in ["conv", "densenet"]:
                # just included for debugging
                transforms.append(
                    ConditionalCoupling(in_channels=sample_shape[0],
                                        num_context=num_context,
                                        num_blocks=num_blocks,
                                        mid_channels=mid_channels,
                                        depth=1,
                                        dropout=dropout,
                                        gated_conv=False,
                                        coupling_network=coupling_network,
                                        checkerboard=checkerboard,
                                        flip=flip))
            
            elif coupling_network == "transformer":
                transforms.append(
                    ConditionalMixtureCoupling(in_channels=sample_shape[0],
                                               num_context=num_context,
                                               mid_channels=mid_channels,
                                               num_mixtures=num_mixtures,
                                               num_blocks=num_blocks,
                                               dropout=dropout,
                                               use_attn=False,
                                               checkerboard=checkerboard,
                                               flip=flip))
            else:
                raise ValueError(f"Unknown network type {coupling_network}")

        # Final shuffle of channels, squeeze and sigmoid
        transforms.extend([Conv1x1(sample_shape[0]),
                           Unsqueeze2d(),
                           Sigmoid()])
        super(AugmentFlow, self).__init__(base_dist=StandardNormal(sample_shape), # ConvNormal2d(sample_shape),
                                          transforms=transforms,
                                          context_init=context_net)
