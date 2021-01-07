import torch
import argparse
import torch.nn as nn

# Optim
from torch.optim import Adam
from survae.utils import elbo_bpd, iwbo_bpd, elbo_nats, iwbo_nats, count_parameters

# Plot
import torchvision.utils as vutils

# Data
from survae.data.loaders.image import DynamicallyBinarizedMNIST, MNIST

# Model
from survae.flows import Flow
from survae.transforms import AffineCouplingBijection, ActNormBijection2d, Conv1x1, LogisticMixtureAffineCouplingBijection
from survae.transforms import UniformDequantization, Squeeze2d, Slice, ScalarAffineBijection, Augment, Logit
from survae.distributions import StandardNormal, StandardUniform, ConvNormal2d
from survae.nn.nets import TransformerNet
from survae.nn.layers import ElementwiseParams2d, scale_fn




############
## Device ##
############

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 50
k = 8

##########
## Data ##
##########

#data = DynamicallyBinarizedMNIST(as_float=False)
data = MNIST(num_bits=8)
train_loader, test_loader = data.get_data_loaders(32)

###########
## Model ##
###########

def net(in_channels):
    return nn.Sequential(TransformerNet(in_channels//2, mid_channels=16, num_blocks=2, num_mixtures=k, dropout=0.2),
                         ElementwiseParams2d(2 + k * 3))


#model = Flow(base_dist=StandardNormal((16,7,7)),
model = Flow(base_dist=ConvNormal2d((16,7,7)),
             transforms=[
                 UniformDequantization(num_bits=8),
                 #Logit(),
                 ScalarAffineBijection(shift=-0.5),
                 Squeeze2d(),
                 ActNormBijection2d(4), Conv1x1(4), LogisticMixtureAffineCouplingBijection(net(4), num_mixtures=k, scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), LogisticMixtureAffineCouplingBijection(net(4), num_mixtures=k, scale_fn=scale_fn("tanh_exp")),
                 Squeeze2d(),
                 ActNormBijection2d(16), Conv1x1(16), LogisticMixtureAffineCouplingBijection(net(16), num_mixtures=k, scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), LogisticMixtureAffineCouplingBijection(net(16), num_mixtures=k, scale_fn=scale_fn("tanh_exp")),
             ]).to(device)

print(model)
print("Number of parameters:", count_parameters(model))

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
        if i == 0 and epoch == 0:
            print("Data:", x.size(), "min = ", x.min().data.item(), "max = ", x.max().data.item())
        
        optimizer.zero_grad()
        #loss = elbo_nats(model, x.to(device))
        loss = elbo_bpd(model, x.to(device))
        #loss = -model.log_prob(x.to(device)).mean()
        loss.backward()
        optimizer.step()
        l += loss.detach().cpu().item()
        print('Epoch: {}/{}, Iter: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, epochs, i+1, len(train_loader), l/(i+1)), end='\r')
    print('')

##########
## Test ##
##########

print('Testing...')
with torch.no_grad():
    l = 0.0
    for i, x in enumerate(test_loader):
        #loss = iwbo_nats(model, x.to(device), k=10)
        loss = iwbo_bpd(model, x.to(device), k=10)
        l += loss.detach().cpu().item()
        print('Iter: {}/{}, Bits/dim: {:.3f}'.format(i+1, len(test_loader), l/(i+1)), end='\r')
    print('')

############
## Sample ##
############

print('Sampling...')
img = next(iter(test_loader))[:64]
samples = model.sample(64)

print("samples:", samples.size(), "min = ", samples.min().data.item(), "max = ", samples.max().data.item())

vutils.save_image(img.cpu().float(), './examples/results/mnist_data.png', nrow=8)
vutils.save_image(samples.cpu().float(), './examples/results/mnist_flowpp.png', nrow=8)
