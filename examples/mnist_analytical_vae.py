import torch
import argparse
import torch.nn as nn
import numpy as np
import math
import torchvision.utils as vutils
from torch.optim import Adam

from utils import set_seeds
from survae.data.loaders.image import DynamicallyBinarizedMNIST
from survae.utils import iwbo_nats
from survae.flows import Flow
from survae.transforms import VAE
from survae.transforms import AffineCouplingBijection, ActNormBijection2d, Conv1x1
from survae.transforms import UniformDequantization, Squeeze2d, Slice, ScalarAffineBijection
from survae.distributions import StandardNormal, ConditionalNormal
from survae.nn.nets import MLP, DenseNet
from survae.nn.layers import ElementwiseParams2d



parser = argparse.ArgumentParser(description='Compare analytical and stochastic VAE.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--stochastic_loss', type)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args(main_args)

set_seeds(args.seed)
LOG_2PI = np.log(2 * np.pi)

stochastic=True
latent_sz = 256
vae_in_sz = 16 * 7 * 7

#
# Linear VAE class with analytical solution
#

class LinearVAE(nn.Module):
    """
    Linear VAE

    Allows for the reconstruction error to be computed analytically or via
    single-sample stochastic estimation.

    Args:
    input_dim: Number of input dimensions (default: 784)
    hidden_dim: Number of hidden dimensions (default: 20)
    trainable_sigma: Whether observation noise should be learned (default: True)
    sigma_init: Initial value of sigma^2 (default: None)
    stochastic_ELBO: Use stochastic ELBO estimation (default: False)
    """
    def __init__(self,
                 input_dim=784,
                 hidden_dim=20,
                 trainable_sigma=True,
                 sigma_init=1.0,
                 stochastic_ELBO=False):

        super(LinearVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.trainable_sigma = trainable_sigma
        self.sigma_init = sigma_init
        self.stochastic_ELBO = stochastic_ELBO

        # define model parameters
        self.q_logvar = nn.Parameter(torch.ones(hidden_dim, requires_grad=True))
        self.enc_weight = nn.Parameter(torch.zeros(input_dim, hidden_dim, requires_grad=True))
        self.mu = nn.Parameter(torch.zeros(input_dim, requires_grad=True))
        self.dec_weight = nn.Parameter(torch.zeros(hidden_dim, input_dim, requires_grad=True))

        if sigma_init is not None:
            self.log_sigma = nn.Parameter(torch.log(sigma_init * torch.ones(1, requires_grad=trainable_sigma)))
        else:
            self.log_sigma = nn.Parameter(torch.ones(1, requires_grad=trainable_sigma))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        batch_size = x.size(0)
        q_mu = torch.matmul(x - self.mu, self.enc_weight)

        if self.stochastic_ELBO:
            std = torch.exp(0.5 * self.q_logvar.unsqueeze(0)).repeat(batch_size, 1)
            eps = torch.FloatTensor(q_mu.size()).normal_()
            z = eps.mul(std).add_(q_mu)
            x_recon = torch.matmul(z, self.dec_weight)
            loss = self.stochastic_loss(x, x_recon)
        else:
            loss = self.analytic_loss(x)

        kl = self.gaussian_kl_divergence(q_mu, p_mean = 0.0, p_logvar=2 * math.log(1.0))
        neg_elbo = loss + kl
        return neg_elbo

    def gaussian_kl_divergence(self, q_mu, p_mean, p_logvar):
        """
        KL Divergence between two Gaussian distributions.

        Given q ~ N(mu_1, sigma^2_1) and p ~ N(mu_2, sigma^2_2), this function
        returns,

        KL(q||p) = log (sigma^2_2 / sigma^2_1) +
        (sigma^2_1 + (mu_1 - mu_2)^2) / (2 sigma^2_2) - 0.5

        Args:
        q_mean: Mean of proposal distribution.
        p_mean: Mean of prior distribution.
        p_logvar: Log-variance of prior distribution

        q_logvar: Log-variance of proposal distribution stored as class parameter

        Returns:
        The KL divergence between q and p ( KL(q||p) ).
        """
        rval = 0.5 * (p_logvar - self.q_logvar +
                      (torch.exp(self.q_logvar) + (q_mu - p_mean)**2) / math.exp(p_logvar) - 1.0)
        # average over batches
        rval = torch.mean(rval, dim=0)
        # sum over all dimensions
        rval = torch.sum(rval)
        return rval

    def analytic_loss(self, x):
        """
        Compute the analytic reconstruction error for a linear VAE.

        Args:
        x: The input tensor.

        Returns:

        E_q[log p(x|z)]
        """
        wv = torch.matmul(self.enc_weight, self.dec_weight)
        x_sub_mu = x - self.mu

        wvx = torch.matmul(x_sub_mu, wv)  # does this mess with batch dim? bmm()?
        xvwwvx = torch.sum(wvx * wvx, dim=1)

        sigma_sq = torch.exp(self.q_logvar.unsqueeze(1))
        tr_wdw = torch.trace(torch.matmul(self.dec_weight.t(), sigma_sq * self.dec_weight))

        xwvx = torch.sum(wvx * x_sub_mu, dim=1)

        xx = torch.sum(x_sub_mu * x_sub_mu, dim=1)

        d = self.input_dim
        rval = torch.mean(
            0.5 * ((tr_wdw + xvwwvx - 2.0 * xwvx + xx) / sigma_sq + d * (LOG_2PI + self.log_sigma)))
        return rval

    def stochastic_loss(self, x, logits):
        """
        Stochastic estimation of the reconstruction error.

        Computes the reconstruction error under a Gaussian observation model.

        Args:
        x: The input tensor.
        logits: The stochastic output of the decoder.

        Returns:

        MEAN_i(log p(x_i|z_i))
        """
        input_dim = x.size(1)
        sigma_sq = torch.exp(self.log_sigma)
        rval = torch.mean(
            0.5 * (torch.sum((x - logits)**2 / sigma_sq, 1) + (LOG_2PI + self.log_sigma) * input_dim))
        return rval


############
## Device ##
############

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##########
## Data ##
##########

data = DynamicallyBinarizedMNIST()
train_loader, test_loader = data.get_data_loaders(args.batch_size)

###########
## Model ##
###########


linvae = LinearVAE(input_dim=vae_in_sz, hidden_dim=latent_sz, trainable_sigma=True, sigma_init=1.0, stochastic_ELBO=stochastic)

def net(channels):
    return nn.Sequential(DenseNet(in_channels=channels//2,
                                  out_channels=channels,
                                  num_blocks=1,
                                  mid_channels=64,
                                  depth=2,
                                  growth=16,
                                  dropout=0.0,
                                  gated_conv=True,
                                  zero_init=True),
                         ElementwiseParams2d(2))

model = Flow(base_dist=StandardNormal((16,7,7)),
                transforms=[
                    UniformDequantization(num_bits=8),
                    ScalarAffineBijection(shift=-0.5),
                    Squeeze2d(),
                    ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4)),
                    Squeeze2d(),
                    ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16))
                ]).to(device)

print(model)
print(linvae)

###########
## Optim ##
###########


optimizer = Adam(list(model.parameters()) + list(linvae.parameters()), lr=1e-3)

###########
## Train ##
###########

print('Training...')
for epoch in range(2):
    l = 0.0
    for i, x in enumerate(train_loader):
        optimizer.zero_grad()
        z, log_prob = model.log_prob(x.to(device), return_z=True)
        neg_elbo = linvae(z)
        loss = neg_elbo - log_prob
        loss.backward()
        optimizer.step()
        l += loss.detach().cpu().item()
        print('Epoch: {}/{}, Iter: {}/{}, ELBO: {:.3f}'.format(epoch+1, 20, i+1, len(train_loader), -1.0 * l/(i+1)), end='\r')
    print('')

##########
## Test ##
##########

# print('Testing...')
# with torch.no_grad():
#     l = 0.0
#     for i, x in enumerate(test_loader):
#         loss = iwbo_nats(model, x.to(device), k=10)
#         l += loss.detach().cpu().item()
#         print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), l/(i+1)), end='\r')
#     print('')

############
## Sample ##
############

# print('Sampling...')
# img = next(iter(test_loader))[:64]
# samples = model.sample(64)

# vutils.save_image(img.cpu().float(), fp='mnist_data.png', nrow=8)
# vutils.save_image(samples.cpu().float(), fp='mnist_ndp_relu.png', nrow=8)
