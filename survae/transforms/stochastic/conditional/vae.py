import torch
from survae.transforms.stochastic.conditional import ConditionalStochasticTransform


class ConditionalVAE(ConditionalStochasticTransform):
    '''
    A conditional variational autoencoder [1, 2] layer.
    Encoder and decoder take an additional context argument that is
    combined with the input x.

    Args:
        decoder: ConditionalDistribution, the decoder p(x|z).
        encoder: ConditionalDistribution, the encoder q(z|x).

    References:
        [1] Auto-Encoding Variational Bayes,
            Kingma & Welling, 2013, https://arxiv.org/abs/1312.6114
        [2] Stochastic Backpropagation and Approximate Inference in Deep Generative Models,
            Rezende et al., 2014, https://arxiv.org/abs/1401.4082
    '''

    def __init__(self, decoder, encoder, context_net=None):
        super(VAE, self).__init__()
        self.context_net = context_net
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, context):
        if self.context_net: context = self.context_net(context)

        if context.shape[2] == 1 and context.shape[3] == 1:
            context = x + context
        else:
            context = torch.cat([x, context], dim=1)

        z, log_qz = self.encoder.sample_with_log_prob(context=context)
        log_px = self.decoder.log_prob(x, context=z)
        return z, log_px - log_qz

    def inverse(self, z, context):
        if self.context_net: context = self.context_net(context)
        if context.shape[2] == 1 and context.shape[3] == 1:
            context = z + context
        else:
            context = torch.cat([z, context], dim=1)
        
        return self.decoder.sample(context=context)
