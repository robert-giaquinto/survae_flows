import torch
import torch.nn as nn
import math

from torch.distributions import Normal

from survae.transforms.stochastic import StochasticTransform
from survae.utils import sum_except_batch


class LinearVAE(StochasticTransform):
    '''
    A linear variational autoencoder [1, 2] layer.
    Parameters can either be learned analytically or through stochastic backprop

    Args:
        input_dim:          Number of input dimensions (default: 784)
        hidden_dim:         Number of hidden dimensions (default: 20)
        trainable_sigma:    Whether observation noise should be learned (default: True)
        sigma_init:         Initial value of sigma^2 (default: None)
        stochastic_elbo:    Use stochastic ELBO estimation (default: False)

    References:
        [1] Auto-Encoding Variational Bayes,
            Kingma & Welling, 2013, https://arxiv.org/abs/1312.6114
        [2] Stochastic Backpropagation and Approximate Inference in Deep Generative Models,
            Rezende et al., 2014, https://arxiv.org/abs/1401.4082
    '''

    def __init__(self, input_dim, hidden_dim, trainable_sigma=True, sigma_init=1.0, stochastic_elbo=False):

        super(LinearVAE, self).__init__()

        self.input_shape = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.trainable_sigma = trainable_sigma
        self.sigma_init = sigma_init
        self.stochastic_elbo = stochastic_elbo

        # define model parameters
        self.z_logvar = nn.Parameter(torch.zeros(hidden_dim, requires_grad=True))
        self.enc_weight = nn.Parameter(torch.zeros(input_dim, hidden_dim, requires_grad=True))  # V
        self.mu = nn.Parameter(torch.zeros(input_dim, requires_grad=True))
        self.dec_weight = nn.Parameter(torch.zeros(hidden_dim, input_dim, requires_grad=True))  # W

        if sigma_init is not None:
            self.log_sigma = nn.Parameter(torch.log(sigma_init * torch.ones(1, requires_grad=trainable_sigma)))
        else:
            self.log_sigma = nn.Parameter(torch.ones(1, requires_grad=trainable_sigma))

    def forward(self, x):
        """
        Encode x to latent z, also return log likelihood p(x|z)
        """
        # reshape to a flattened vector
        self.input_shape = x.size()
        x = x.view(x.size(0), -1)
        
        # Project x to z
        z_mean = torch.matmul(x - self.mu, self.enc_weight)
        z_std = torch.exp(0.5 * self.z_logvar.unsqueeze(0)).repeat(x.size(0), 1)
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        eps = FloatTensor(z_mean.size()).normal_()
        z = eps.mul(z_std).add_(z_mean)

        if self.stochastic_elbo or self.training == False:
            x_recon = torch.matmul(z, self.dec_weight)
            lhood = self.stochastic_lhood(x, x_recon)
        else:
            lhood = self.analytic_lhood(x)

        #kl = self.gaussian_kl_divergence(z_mean, self.z_logvar, p_mean=0.0, p_logvar=math.log(1.0))
        #elbo = lhood - kl
        return z, lhood

    def inverse(self, z):
        """
        Decode latent z into data x
        """
        x_mean = torch.matmul(z, self.dec_weight)
        x_sigma = torch.exp(self.log_sigma)
        decoder =  Normal(loc=x_mean, scale=x_sigma)
        x_recon = decoder.rsample().view((z.size(0), self.input_shape[1], self.input_shape[2], self.input_shape[3]))
        return x_recon

    def gaussian_kl_divergence(self, q_mean, q_logvar, p_mean, p_logvar):
        """
        KL Divergence between two Gaussian distributions.

        Given q ~ N(mu_1, sigma^2_1) and p ~ N(mu_2, sigma^2_2), this function
        returns,

        KL(q||p) = log (sigma^2_2 / sigma^2_1) +
        (sigma^2_1 + (mu_1 - mu_2)^2) / (2 sigma^2_2) - 0.5

        Args:
        q_mean: Mean of proposal distribution.
        q_logvar: Log-variance of proposal distribution.
        p_mean: Mean of prior distribution.
        p_logvar: Log-variance of prior distribution

        Returns: The KL divergence between q and p ( KL(q||p) ).
        """
        rval = (p_logvar - q_logvar) + \
            0.5 * ((torch.exp(q_logvar) + (q_mean - p_mean)**2) / math.exp(p_logvar) - 1.0)
        return sum_except_batch(rval)

    def analytic_lhood(self, x):
        """
        Compute the analytic reconstruction error for a linear VAE.

        Args:
            x: The input tensor.

        Returns: E_q[log p(x|z)]
        """
        wv = torch.matmul(self.enc_weight, self.dec_weight)
        x_sub_mu = x - self.mu

        wvx = torch.matmul(x_sub_mu, wv)
        xvwwvx = torch.sum(wvx * wvx, dim=1)
        z_var = torch.exp(self.z_logvar.unsqueeze(1))
        tr_wdw = torch.trace(torch.matmul(self.dec_weight.t(), z_var * self.dec_weight))
        xwvx = torch.sum(wvx * x_sub_mu, dim=1)
        xx = torch.sum(x_sub_mu * x_sub_mu, dim=1)

        d = self.input_dim
        sigma_sq = torch.exp(self.log_sigma)
        log_inner = -0.5 * (tr_wdw + xvwwvx - 2.0 * xwvx + xx) / sigma_sq
        log_base = -0.5 * d * (math.log(2 * math.pi) + self.log_sigma)
        rval = sum_except_batch(log_base + log_inner)
        return rval

    def stochastic_lhood(self, x, logits):
        """
        Stochastic estimation of the reconstruction error.

        Computes the reconstruction error under a Gaussian observation model.

        Args:
            x: The input tensor.
            logits: The stochastic output of the decoder.

        Returns: Mean_i(log p(x_i|z_i))
        """
        #input_dim = x.size(1)
        d = self.input_dim
        assert d == x.size(1)
        sigma_sq = torch.exp(self.log_sigma)
        log_inner = -0.5 * torch.sum((x - logits)**2 / sigma_sq, 1)
        log_base = -0.5 * d * (math.log(2 * math.pi) + self.log_sigma)
        rval = sum_except_batch(log_base + log_inner)
        return rval



