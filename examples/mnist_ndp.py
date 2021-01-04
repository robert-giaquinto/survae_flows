import torch
import argparse
import torch.nn as nn

# Optim
from torch.optim import Adam
from survae.utils import iwbo_nats

# Plot
import torchvision.utils as vutils

# Data
from survae.data.loaders.image import DynamicallyBinarizedMNIST

# Model
from survae.flows import NDPFlow
from survae.transforms import VAE
from survae.transforms import AffineCouplingBijection, ActNormBijection2d, Conv1x1
from survae.transforms import UniformDequantization, Squeeze2d, Slice, ScalarAffineBijection
from survae.distributions import StandardNormal, ConditionalNormal
from survae.nn.nets import MLP, DenseNet, ConvNet
from survae.nn.layers import ElementwiseParams2d




############
## Device ##
############

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 15
latent_size = 392
vae_in_size = 16 * 7 * 7


##########
## Data ##
##########

data = DynamicallyBinarizedMNIST()
train_loader, test_loader = data.get_data_loaders(128)

###########
## Model ##
###########


encoder = ConditionalNormal(MLP(vae_in_size, 2 * latent_size,
                                hidden_units=[512, 256],
                                activation='relu',
                                in_lambda=lambda x: x.view(x.shape[0], vae_in_size)))
decoder = ConditionalNormal(MLP(latent_size, vae_in_size * 2,
                                hidden_units=[256, 512],
                                activation='relu',
                                out_lambda=lambda x: x.view(x.shape[0], 32, 7, 7)), split_dim=1)


use_conv_coupling = True
if use_conv_coupling:
    def net(channels):
        return nn.Sequential(ConvNet(input_size=channels//2,
                                     output_size=channels,
                                     hidden_units=64,
                                     num_layers=1,
                                     activation='relu'),
                             ElementwiseParams2d(2))

else:
    def net(channels):
        return nn.Sequential(DenseNet(in_channels=channels//2,
                                      out_channels=channels,
                                      num_blocks=1,
                                      mid_channels=64,
                                      depth=1,
                                      growth=16,
                                      dropout=0.0,
                                      gated_conv=True,
                                      zero_init=True),
                             ElementwiseParams2d(2))



#model = NDPFlow(base_dist=[StandardNormal((16,7,7)), StandardNormal((latent_size,))],
model = NDPFlow(base_dist=[None, StandardNormal((latent_size,))],
                transforms=[
                    UniformDequantization(num_bits=8),
                    ActNormBijection2d(1),
                    ScalarAffineBijection(scale=2.0),
                    ScalarAffineBijection(shift=-0.5),
                    Squeeze2d(),
                    ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4)),
                    ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4)),
                    Squeeze2d(),
                    ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16)),
                    ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16)),
                    VAE(encoder=encoder, decoder=decoder)
                ]).to(device)

print(model)

###########
## Optim ##
###########

optimizer = Adam(model.parameters(), lr=1e-3)

###########
## Train ##
###########

print('Training...')
for epoch in range(epochs):
    l = 0.0
    for i, x in enumerate(train_loader):
        optimizer.zero_grad()
        loss = -model.log_prob(x.to(device)).mean()
        loss.backward()
        optimizer.step()
        l += loss.detach().cpu().item()
        print('Epoch: {}/{}, Iter: {}/{}, Nats: {:.3f}'.format(epoch+1, epochs, i+1, len(train_loader), l/(i+1)), end='\r')
    print('')

##########
## Test ##
##########

print('Testing...')
with torch.no_grad():
    l = 0.0
    for i, x in enumerate(test_loader):
        loss = iwbo_nats(model, x.to(device), k=10)
        l += loss.detach().cpu().item()
        print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), l/(i+1)), end='\r')
    print('')

############
## Sample ##
############

print('Sampling...')
img = next(iter(test_loader))[:64]
samples = model.sample(64)

vutils.save_image(img.cpu().float(), 'mnist_data.png', nrow=8)
vutils.save_image(samples.cpu().float(), 'mnist_ndp_392.png', nrow=8)
