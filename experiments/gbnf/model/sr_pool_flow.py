import torch
import torch.nn as nn
from survae.flows import ConditionalFlow
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import ScalarAffineBijection, Squeeze2d, Conv1x1, Slice, SimpleMaxPoolSurjection2d, ActNormBijection2d
from survae.transforms import ConditionalConv1x1, ConditionalActNormBijection2d
from survae.nn.nets import DenseNet, ConvDecoderNet

from model.coupling import SRCoupling, SRMixtureCoupling
from model.dequantization_flow import DequantizationFlow


class ContextInit(nn.Module):
    def __init__(self, num_bits, context_shape, out_shape=None, num_blocks=1, depth=1, mid_channels=[16], dropout=0.0, norm=False):
        super(ContextInit, self).__init__()
        self.dequant = UniformDequantization(num_bits=num_bits)
        self.shift = ScalarAffineBijection(shift=-0.5)

        mid_channels = list(mid_channels)
        self.encode, self.upsample = None, None
        if len(mid_channels) > 0:
            if out_shape is None:
                out_shape = context_shape

            self.encode = DenseNet(in_channels=context_shape[0],
                                    out_channels=context_shape[0],
                                    num_blocks=num_blocks,
                                    mid_channels=min(mid_channels),
                                    depth=depth,
                                    growth=min(mid_channels),
                                    dropout=dropout,
                                    gated_conv=False,
                                    zero_init=False)
            self.upsample = ConvDecoderNet(in_channels=context_shape[0],
                                           out_shape=out_shape,
                                           mid_channels=mid_channels,
                                           batch_norm=norm)
        
    def forward(self, context):
        context, _ = self.dequant(context)
        context, _ = self.shift(context)
        if self.encode:
            #context = self.encode(context)
            context = self.upsample(context)
            
        return context


class SRPoolFlow(ConditionalFlow):

    def __init__(self, data_shape, cond_shape, num_bits,
                 num_scales, num_steps,
                 actnorm, conditional_channels,
                 pooling,
                 coupling_network,
                 dequant, dequant_steps, dequant_context,
                 coupling_blocks, coupling_channels, coupling_dropout,
                 coupling_gated_conv=None, coupling_depth=None, coupling_mixtures=None):

        batch_norm=False
        conditional_channels = [coupling_channels // 2] if conditional_channels is None else list(conditional_channels)
        
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

        # Pooling flows
        for scale in range(num_scales):
            for step in range(num_steps):

                if len(conditional_channels) == 0:
                    if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                    transforms.append(Conv1x1(current_shape[0]))
                else:
                    if actnorm: transforms.append(ConditionalActNormBijection2d(cond_shape=cond_shape,
                                                                                out_channels=current_shape[0],
                                                                                mid_channels=conditional_channels, norm=batch_norm))
                    transforms.append(ConditionalConv1x1(cond_shape=cond_shape,
                                                         out_channels=current_shape[0],
                                                         mid_channels=conditional_channels, norm=batch_norm))
                    
                if coupling_network in ["conv", "densenet"]:
                    transforms.append(SRCoupling(x_size=cond_shape, 
                                                 y_size=current_shape,
                                                 mid_channels=coupling_channels,
                                                 depth=coupling_depth,
                                                 norm=batch_norm,
                                                 coupling_network=coupling_network))
                elif coupling_network == "transformer":
                    transforms.append(SRMixtureCoupling(x_size=cond_shape, 
                                                        y_size=current_shape,
                                                        mid_channels=coupling_channels,
                                                        dropout=coupling_dropout,
                                                        num_blocks=coupling_blocks,
                                                        num_mixtures=coupling_mixtures,
                                                        norm=batch_norm))


            if scale < num_scales-1:
                if pooling == 'none':
                    transforms.append(Squeeze2d())
                    current_shape = (current_shape[0] * 4,
                                     current_shape[1] // 2,
                                     current_shape[2] // 2)
                else:
                
                    if pooling == 'slice':
                        noise_shape = (current_shape[0] * 2,
                                       current_shape[1] // 2,
                                       current_shape[2] // 2)
                        transforms.append(Squeeze2d())
                        transforms.append(Slice(StandardNormal(noise_shape), num_keep=current_shape[0] * 2, dim=1))
                        current_shape = (current_shape[0] * 2,
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
                        raise ValueError("pooling argument must be either slice, max or none")
                
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))

        # for reference save the shape output by the bijective flow
        self.flow_shape = current_shape

        context_init = ContextInit(num_bits=num_bits, context_shape=cond_shape, out_shape=data_shape,
                                   num_blocks=1,
                                   depth=1,
                                   mid_channels=[coupling_channels // 2],
                                   dropout=0.0,
                                   norm=batch_norm)
        super(SRPoolFlow, self).__init__(base_dist=ConvNormal2d(current_shape),
                                         transforms=transforms,
                                         context_init=context_init)
