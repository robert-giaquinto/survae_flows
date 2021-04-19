import torch
import torch.nn as nn

from survae.flows import ConditionalInverseFlow
from survae.distributions import ConvNormal2d
from survae.transforms import Unsqueeze2d, Sigmoid, Conv1x1, ActNormBijection2d
from survae.nn.layers import LambdaLayer, Conv2d, Conv2dZeros, Checkerboard
from survae.nn.blocks import DenseBlock
from survae.nn.nets import ConvNet, GatedConvNet, TransformerNet, ConvAttnBlock

from .coupling import ConditionalCoupling, ConditionalMixtureCoupling


class DequantizationFlow(ConditionalInverseFlow):

    def __init__(self, data_shape, num_bits, num_steps, coupling_network, num_context,
                 mid_channels, num_blocks=4, depth=4, growth=32, dropout=0.0, gated_conv=False, num_mixtures=4, checkerboard=True):

        context_in = data_shape[0] * 2 if checkerboard else data_shape[0]
        context_network_type = "conv"

        # define the context net to be used
        if context_network_type == "densenet":
            layers = [LambdaLayer(lambda x: 2*x.float()/(2**num_bits-1)-1)]            
            if checkerboard: layers.append(Checkerboard(concat_dim=1))
            layers += [DenseBlock(in_channels=context_in, out_channels=mid_channels,
                                  depth=depth, growth=growth, dropout=dropout, gated_conv=gated_conv, zero_init=False),
                       nn.Conv2d(mid_channels, mid_channels, kernel_size=2, stride=2, padding=0),
                       DenseBlock(in_channels=mid_channels, out_channels=num_context,
                                  depth=depth, growth=growth, dropout=dropout, gated_conv=gated_conv, zero_init=False)]
            
            context_net = nn.Sequential(*layers)
            
        elif context_network_type == "transformer":
            layers = [LambdaLayer(lambda x: 2*x.float()/(2**num_bits-1)-1)]
            if checkerboard: layers.append(Checkerboard(concat_dim=1))
            layers += [Conv2d(in_channels=context_in, out_channels=mid_channels, kernel_size=3, padding=1),
                       nn.Conv2d(mid_channels, mid_channels, kernel_size=2, stride=2, padding=0)]
            
            for i in range(num_blocks):
                layers.append(ConvAttnBlock(channels=mid_channels, dropout=0.0, use_attn=False, context_channels=None))
                
            layers.append(Conv2d(in_channels=mid_channels, out_channels=num_context, kernel_size=3, padding=1))
            context_net = nn.Sequential(*layers)

        elif context_network_type == "conv":
            layers = [LambdaLayer(lambda x: 2*x.float()/(2**num_bits-1)-1)]
            
            if checkerboard: layers.append(Checkerboard(concat_dim=1))

            layers += [Conv2d(context_in, mid_channels // 2, kernel_size=3, stride=1),
                       nn.Conv2d(mid_channels // 2, mid_channels, kernel_size=2, stride=2, padding=0),
                       GatedConvNet(channels=mid_channels, num_blocks=2, dropout=0.0),
                       Conv2dZeros(in_channels=mid_channels, out_channels=num_context)]

            context_net = nn.Sequential(*layers)
            
        else:
            raise ValueError(f"Unknown dequantization context_network_type type: {context_network_type}")

        
        # layer transformations of the dequantization flow
        transforms = []
        sample_shape = (data_shape[0] * 4, data_shape[1] // 2, data_shape[2] // 2)
        for i in range(num_steps):
            transforms.append(ActNormBijection2d(sample_shape[0]))
            transforms.extend([Conv1x1(sample_shape[0])])
            if coupling_network in ["conv", "densenet"]:
                transforms.append(
                    ConditionalCoupling(in_channels=sample_shape[0],
                                        num_context=num_context,
                                        num_blocks=num_blocks,
                                        mid_channels=mid_channels,
                                        depth=depth,
                                        dropout=dropout,
                                        gated_conv=gated_conv,
                                        coupling_network=coupling_network,
                                        checkerboard=checkerboard,
                                        flip=(i % 2 == 0)))
                
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
                                               flip=(i % 2 == 0)))
            else:
                raise ValueError(f"Unknown network type {coupling_network}")

        # Final shuffle of channels, squeeze and sigmoid
        transforms.extend([Conv1x1(sample_shape[0]),
                           Unsqueeze2d(),
                           Sigmoid()])
        super(DequantizationFlow, self).__init__(base_dist=ConvNormal2d(sample_shape),
                                                 transforms=transforms,
                                                 context_init=context_net)

