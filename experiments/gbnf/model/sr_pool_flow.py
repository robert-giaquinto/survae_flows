import torch
import torch.nn as nn
from survae.flows import ConditionalFlow
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal, ConditionalNormal, StandardUniform
from survae.transforms import VAE, ConditionalVAE, Reshape
from survae.transforms import UniformDequantization, VariationalDequantization, Augment
from survae.transforms import ScalarAffineBijection, Squeeze2d, Conv1x1, Slice, SimpleMaxPoolSurjection2d, ActNormBijection2d
from survae.transforms import ConditionalConv1x1, ConditionalActNormBijection2d
from survae.nn.nets import ConvEncoderNet, ConvDecoderNet, DenseNet, ContextUpsampler, UpsamplerNet

from model.coupling import SRCoupling, SRMixtureCoupling, Coupling, MixtureCoupling
from model.dequantization_flow import DequantizationFlow
from model.augment_flow import AugmentFlow


class ContextInit(nn.Module):
    def __init__(self, num_bits, in_channels, out_channels, mid_channels, num_blocks, depth, dropout=0.0):
        super(ContextInit, self).__init__()
        self.dequant = UniformDequantization(num_bits=num_bits)
        self.shift = ScalarAffineBijection(shift=-0.5)

        self.encode = None
        if mid_channels > 0 and num_blocks > 0 and depth > 0:
            self.encode = DenseNet(in_channels=in_channels,
                                   out_channels=out_channels,
                                   num_blocks=num_blocks,
                                   mid_channels=mid_channels,
                                   depth=depth,
                                   growth=mid_channels,
                                   dropout=dropout,
                                   gated_conv=False,
                                   zero_init=False)
        
    def forward(self, context):
        context, _ = self.dequant(context)
        context, _ = self.shift(context)
        if self.encode:
            context = self.encode(context)
            
        return context


class SRPoolFlow(ConditionalFlow):

    def __init__(self, data_shape, cond_shape, num_bits,
                 num_scales, num_steps,
                 actnorm, conditional_channels,
                 lowres_encoder_channels, lowres_encoder_blocks, lowres_encoder_depth, lowres_upsampler_channels,
                 pooling, compression_ratio,
                 coupling_network,
                 coupling_blocks, coupling_channels,
                 coupling_dropout=0.0, coupling_gated_conv=None, coupling_depth=None, coupling_mixtures=None,
                 dequant="flow", dequant_steps=4, dequant_context=32, dequant_blocks=2,
                 augment_steps=4, augment_context=32, augment_blocks=2,
                 augment_size=None, checkerboard_scales=[], tuple_flip=True):

        if len(compression_ratio) == 1 and num_scales > 1:
            compression_ratio = [compression_ratio[0]] * (num_scales - 1)
        assert all([compression_ratio[s] >= 0.0 and compression_ratio[s] < 1.0 for s in range(num_scales-1)])
        
        # initialize context. Only upsample context in ContextInit if latent shape doesn't change during the flow.
        context_init = ContextInit(num_bits=num_bits,
                                   in_channels=cond_shape[0],
                                   out_channels=lowres_encoder_channels,
                                   mid_channels=lowres_encoder_channels,
                                   num_blocks=lowres_encoder_blocks,
                                   depth=lowres_encoder_depth,
                                   dropout=coupling_dropout)
        
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
            #transforms.append(Augment(StandardUniform((augment_size, current_shape[1], current_shape[2])), x_size=current_shape[0]))
            #transforms.append(Augment(StandardNormal((augment_size, current_shape[1], current_shape[2])), x_size=current_shape[0]))
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

            # First and Third scales use checkerboard split pattern
            checkerboard = scale in checkerboard_scales
            context_out_channels = min(current_shape[0], coupling_channels)
            context_out_shape = (context_out_channels, current_shape[1], current_shape[2] // 2) if checkerboard else (context_out_channels, current_shape[1], current_shape[2])

            # reshape the context to the current size for all flow steps at this scale
            context_upsampler_net = UpsamplerNet(in_channels=lowres_encoder_channels, out_shape=context_out_shape, mid_channels=lowres_upsampler_channels)
            transforms.append(ContextUpsampler(context_net=context_upsampler_net, direction='forward'))    
            
            for step in range(num_steps):

                flip = (step % 2 == 0) if tuple_flip else False
                
                if len(conditional_channels) == 0:
                    if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))
                    transforms.append(Conv1x1(current_shape[0]))
                else:
                    if actnorm: transforms.append(ConditionalActNormBijection2d(cond_shape=current_shape, out_channels=current_shape[0], mid_channels=conditional_channels))
                    transforms.append(ConditionalConv1x1(cond_shape=current_shape, out_channels=current_shape[0], mid_channels=conditional_channels, slogdet_cpu=True))

                if coupling_network in ["conv", "densenet"]:
                    transforms.append(SRCoupling(x_size=context_out_shape, 
                                                 y_size=current_shape,
                                                 mid_channels=coupling_channels,
                                                 depth=coupling_depth,
                                                 num_blocks=coupling_blocks,
                                                 dropout=coupling_dropout,
                                                 gated_conv=coupling_gated_conv,
                                                 coupling_network=coupling_network,
                                                 checkerboard=checkerboard,
                                                 flip=flip))
                    
                elif coupling_network == "transformer":
                    transforms.append(SRMixtureCoupling(x_size=context_out_shape,
                                                        y_size=current_shape,
                                                        mid_channels=coupling_channels,
                                                        dropout=coupling_dropout,
                                                        num_blocks=coupling_blocks,
                                                        num_mixtures=coupling_mixtures,
                                                        checkerboard=checkerboard,
                                                        flip=flip))

            # Upsample context (for the previous flows, only if moving in the inverse direction)
            transforms.append(ContextUpsampler(context_net=context_upsampler_net, direction='inverse'))
            
            if scale < num_scales-1:
                if pooling == 'none' or compression_ratio[scale] == 0.0:
                    # fully bijective flow with multi-scale architecture
                    transforms.append(Squeeze2d())
                    current_shape = (current_shape[0] * 4,
                                     current_shape[1] // 2,
                                     current_shape[2] // 2)
                elif pooling == 'slice':
                    # slice some of the dimensions (channel-wise) out from further flow steps
                    unsliced_channels = int(max(1, 4  * current_shape[0] * (1.0 - compression_ratio[scale])))
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
                    # max pooling to compress dimensions spatially, h//2 and w//2
                    noise_shape = (current_shape[0] * 3,
                                   current_shape[1] // 2,
                                   current_shape[2] // 2)
                    decoder = StandardHalfNormal(noise_shape)
                    transforms.append(SimpleMaxPoolSurjection2d(decoder=decoder))
                    current_shape = (current_shape[0],
                                     current_shape[1] // 2,
                                     current_shape[2] // 2)
                elif pooling == "mvae":
                    # Compressive flow: reduce the dimensionality of data by 2 (channel-wise)
                    compressed_channels = max(1, int(current_shape[0] * (1.0 - compression_ratio[scale])))
                    latent_size = compressed_channels * current_shape[1] * current_shape[2]
                    vae_channels = [current_shape[0] * 2, current_shape[0] * 4, current_shape[0] * 8]
                    encoder = ConditionalNormal(
                        ConvEncoderNet(in_channels=current_shape[0],
                                       out_channels=latent_size,
                                       mid_channels=vae_channels,
                                       max_pool=True,
                                       batch_norm=True), split_dim=1)
                    decoder = ConditionalNormal(
                        ConvDecoderNet(in_channels=latent_size,
                                       out_shape=(current_shape[0] * 2, current_shape[1], current_shape[2]),
                                       mid_channels=list(reversed(vae_channels)),
                                       batch_norm=True,
                                       in_lambda=lambda x: x.view(x.shape[0], x.shape[1], 1, 1)), split_dim=1)
                    transforms.append(VAE(encoder=encoder, decoder=decoder))
                    transforms.append(Reshape(input_shape=(latent_size,), output_shape=(compressed_channels, current_shape[1], current_shape[2])))

                    # after reducing channels with mvae, squeeze to reshape latent space before another sequence of flows
                    transforms.append(Squeeze2d())
                    current_shape = (compressed_channels * 4,  # current_shape[0] * 4
                                     current_shape[1] // 2,
                                     current_shape[2] // 2)

                else:
                    raise ValueError("pooling argument must be either mvae, slice, max, or none")
                
            else:
                if actnorm: transforms.append(ActNormBijection2d(current_shape[0]))

        # for reference save the shape output by the bijective flow
        self.latent_size = current_shape[0] * current_shape[1] * current_shape[2]
        self.flow_shape = current_shape
        
        super(SRPoolFlow, self).__init__(base_dist=ConvNormal2d(current_shape), transforms=transforms, context_init=context_init)
