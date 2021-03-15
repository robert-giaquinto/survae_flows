import torch
import torch.nn as nn

from survae.flows import CompressiveFlow
from survae.transforms import VAE, Reshape
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import AffineCouplingBijection, ScalarAffineBijection
from survae.transforms import Squeeze2d, Conv1x1, Slice, ActNormBijection2d
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal, ConditionalNormal, StandardUniform
from survae.nn.nets import ConvEncoderNet, ConvDecoderNet
from survae.nn.nets import MLP

from .dequantization_flow import DequantizationFlow
from .coupling import Coupling, MixtureCoupling


class CompressPretrained(CompressiveFlow):

    def __init__(self, pretrained_model, latent_size):
        self.flow_shape = pretrained_model.base_dist.shape
        self.latent_size = latent_size

        # initialize transforms with first scale of pretrained models
        transforms = pretrained_model.transforms[0:28]

        # Replace slice layer with a compression flow
        mencoder = ConditionalNormal(
            ConvEncoderNet(in_channels=48,
                           out_channels=768,
                           mid_channels=[64, 128, 256],
                           max_pool=True,
                           batch_norm=True), split_dim=1)
        mdecoder = ConditionalNormal(
            ConvDecoderNet(in_channels=768,
                           out_shape=(48 * 2, 8, 8),
                           mid_channels=[256, 128, 64],
                           batch_norm=True,
                           in_lambda=lambda x: x.view(x.shape[0], x.shape[1], 1, 1)), split_dim=1)

        mid_vae = VAE(encoder=mencoder, decoder=mdecoder)
        reshape = Reshape(input_shape=(768,), output_shape=(12,8,8))
        transforms.extend([mid_vae, reshape])

        # Non-dimension preserving flows
        current_shape = pretrained_model.base_dist.shape
        flat_dim = current_shape[0] * current_shape[1] * current_shape[2]
        fencoder = ConditionalNormal(MLP(flat_dim, 2 * latent_size,
                                        hidden_units=[512, 256],
                                        activation='relu',
                                        in_lambda=lambda x: x.view(x.shape[0], flat_dim)))
        fdecoder = ConditionalNormal(MLP(latent_size, 2 * flat_dim,
                                        hidden_units=[256, 512],
                                        activation='relu',
                                        out_lambda=lambda x: x.view(x.shape[0], current_shape[0]*2, current_shape[1], current_shape[2])), split_dim=1)

        # append last scale of pretrained model and extend with the compressive VAE
        transforms.extend(pretrained_model.transforms[29:])
        final_vae = VAE(encoder=fencoder, decoder=fdecoder)
        transforms.append(final_vae)

        # Base distribution for non-dimension preserving portion of flow
        base1 = StandardNormal((latent_size,))

        super(CompressPretrained, self).__init__(base_dist=[None, base1], transforms=transforms)
