import torch
import torch.nn as nn
from survae.flows import Flow, ConditionalInverseFlow
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import ScalarAffineBijection, Unsqueeze2d, Squeeze2d, Conv1x1, Slice, SimpleMaxPoolSurjection2d, ActNormBijection2d, Sigmoid
from survae.transforms import AffineCouplingBijection, ConditionalAffineCouplingBijection
from survae.utils import sum_except_batch
from survae.nn.nets import DenseNet
from survae.nn.blocks import DenseBlock
from survae.nn.layers import ElementwiseParams2d, LambdaLayer


class PretrainedFlow(Flow):
    """
    Creates structure of flow mimicing the pretrained models available on:

    https://github.com/didriknielsen/survae_flows/releases/tag/v1.0.0

    """
    def __init__(self, data_shape, num_bits, num_scales, num_steps, actnorm, pooling,
                 dequant, dequant_steps, dequant_context,
                 densenet_blocks, densenet_channels, densenet_depth,
                 densenet_growth, dropout, gated_conv):

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

        # Pooling flows
        for scale in range(num_scales):
            for step in range(num_steps):
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                transforms.extend([
                    Conv1x1(current_shape[0]),
                    Coupling(in_channels=current_shape[0],
                             num_blocks=densenet_blocks,
                             mid_channels=densenet_channels,
                             depth=densenet_depth,
                             growth=densenet_growth,
                             dropout=dropout,
                             gated_conv=gated_conv)
                ])

            if scale < num_scales-1:
                noise_shape = (current_shape[0] * 3,
                               current_shape[1] // 2,
                               current_shape[2] // 2)
                if pooling=='none':
                    transforms.append(Squeeze2d())
                    transforms.append(Slice(StandardNormal(noise_shape), num_keep=current_shape[0], dim=1))
                elif pooling=='max':
                    decoder = StandardHalfNormal(noise_shape)
                    transforms.append(SimpleMaxPoolSurjection2d(decoder=decoder))
                current_shape = (current_shape[0],
                                 current_shape[1] // 2,
                                 current_shape[2] // 2)
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))

        super(PretrainedFlow, self).__init__(base_dist=ConvNormal2d(current_shape),
                                       transforms=transforms)



class Coupling(AffineCouplingBijection):

    def __init__(self, in_channels, num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        assert in_channels % 2 == 0

        net = nn.Sequential(DenseNet(in_channels=in_channels//2,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True),
                            ElementwiseParams2d(2, mode='sequential'))
        super(Coupling, self).__init__(coupling_net=net)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        z = shift + torch.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        x = (z - shift) * torch.exp(-log_scale)
        return x


class ConditionalCoupling(ConditionalAffineCouplingBijection):

    def __init__(self, in_channels, num_context, num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        assert in_channels % 2 == 0

        net = nn.Sequential(DenseNet(in_channels=in_channels//2+num_context,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True),
                            ElementwiseParams2d(2, mode='sequential'))
        super(ConditionalCoupling, self).__init__(coupling_net=net)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        z = shift + torch.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        x = (z - shift) * torch.exp(-log_scale)
        return x


class DequantizationFlow(ConditionalInverseFlow):

    def __init__(self, data_shape, num_bits, num_steps, num_context,
                 num_blocks, mid_channels, depth, growth, dropout, gated_conv):

        context_net = nn.Sequential(LambdaLayer(lambda x: 2*x.float()/(2**num_bits-1)-1),
                                    DenseBlock(in_channels=data_shape[0],
                                               out_channels=mid_channels,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False),
                                    nn.Conv2d(mid_channels, mid_channels, kernel_size=2, stride=2, padding=0),
                                    DenseBlock(in_channels=mid_channels,
                                               out_channels=num_context,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False))

        transforms = []
        sample_shape = (data_shape[0] * 4, data_shape[1] // 2, data_shape[2] // 2)
        for i in range(num_steps):
            transforms.extend([
                Conv1x1(sample_shape[0]),
                ConditionalCoupling(in_channels=sample_shape[0],
                                    num_context=num_context,
                                    num_blocks=num_blocks,
                                    mid_channels=mid_channels,
                                    depth=depth,
                                    growth=growth,
                                    dropout=dropout,
                                    gated_conv=gated_conv)
            ])

        # Final shuffle of channels, squeeze and sigmoid
        transforms.extend([Conv1x1(sample_shape[0]),
                           Unsqueeze2d(),
                           Sigmoid()
                          ])

        super(DequantizationFlow, self).__init__(base_dist=ConvNormal2d(sample_shape),
                                                 transforms=transforms,
                                                 context_init=context_net)
