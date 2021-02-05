import torch
import torch.nn as nn

from survae.flows import NDPFlow
from survae.transforms import VAE
from survae.transforms import UniformDequantization, VariationalDequantization
from survae.transforms import AffineCouplingBijection, ScalarAffineBijection
from survae.transforms import Squeeze2d, Conv1x1, Slice, ActNormBijection2d
from survae.distributions import ConvNormal2d, StandardNormal, StandardHalfNormal, ConditionalNormal, StandardUniform
from survae.nn.nets import MLP

from .dequantization_flow import DequantizationFlow
from .coupling import Coupling, MixtureCoupling


class CompressPretrained(NDPFlow):

    def __init__(self, pretrained_model, vae_hidden_units, latent_size, vae_activation):
        self.flow_shape = pretrained_model.base_dist.shape
        self.latent_size = latent_size

        # Initialize flow with pretrained bijective flow
        transforms = pretrained_model.transforms
        current_shape = pretrained_model.base_dist.shape

        # TODO: can have option to "regularize" the samples to be gaussian prior to the compression
        base0 = None

        # Non-dimension preserving flows
        flat_dim = current_shape[0] * current_shape[1] * current_shape[2]
        encoder = ConditionalNormal(MLP(flat_dim, 2 * latent_size,
                                        hidden_units=vae_hidden_units,
                                        activation=vae_activation,
                                        in_lambda=lambda x: x.view(x.shape[0], flat_dim)))
        decoder = ConditionalNormal(MLP(latent_size, 2 * flat_dim,
                                        hidden_units=list(reversed(vae_hidden_units)),
                                        activation=vae_activation,
                                        out_lambda=lambda x: x.view(x.shape[0], current_shape[0]*2, current_shape[1], current_shape[2])), split_dim=1)
        
        transforms.append(VAE(encoder=encoder, decoder=decoder))

        # Base distribution for non-dimension preserving portion of flow
        base1 = StandardNormal((latent_size,))

        super(CompressPretrained, self).__init__(base_dist=[base0, base1], transforms=transforms)
