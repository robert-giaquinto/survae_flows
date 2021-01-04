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
from survae.transforms import AffineCouplingBijection, ActNormBijection2d, Conv1x1
from survae.transforms import UniformDequantization, Squeeze2d, Slice, ScalarAffineBijection, Augment, Logit
from survae.distributions import StandardNormal, StandardUniform, ConvNormal2d
from survae.nn.nets import DenseNet, ConvNet, TransformerNet
from survae.nn.layers import ElementwiseParams2d, scale_fn




############
## Device ##
############

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100
k = 32  # number of components in transformer mixture

##########
## Data ##
##########

#data = DynamicallyBinarizedMNIST(as_float=True)
data = MNIST(num_bits=8)
train_loader, test_loader = data.get_data_loaders(32)

###########
## Model ##
###########

coupling = "transformer"
if coupling == "conv":
    def net(channels):
        return nn.Sequential(ConvNet(in_channels=channels//2,
                                     out_channels=(channels - channels//2) * 2,
                                     mid_channels=64,
                                     num_layers=1,
                                     activation='relu'),
                             ElementwiseParams2d(2))

elif coupling == "transformer":
    def net(channels):
    return nn.Sequential(TransformerNet(channels//2, mid_channels=32, num_blocks=1, num_mixtures=k, dropout=0.2),
                         ElementwiseParams2d(2 + k * 3))
    
elif coupling == "densenet":
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

#model = Flow(base_dist=StandardNormal((16,7,7)),
model = Flow(base_dist=ConvNormal2d((16,7,7)),
             transforms=[
                 UniformDequantization(num_bits=8),
                 # Augment(StandardUniform((1, 28, 28)), x_size=1),
                 # AffineCouplingBijection(net(2)), ActNormBijection2d(2), Conv1x1(2),
                 # AffineCouplingBijection(net(2)), ActNormBijection2d(2), Conv1x1(2),
                 # Squeeze2d(), Slice(StandardNormal((4, 14, 14)), num_keep=4),
                 # AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4),
                 # AffineCouplingBijection(net(4)), ActNormBijection2d(4), Conv1x1(4),
                 # Squeeze2d(), Slice(StandardNormal((8, 7, 7)), num_keep=8),
                 # AffineCouplingBijection(net(8)), ActNormBijection2d(8), Conv1x1(8),
                 # AffineCouplingBijection(net(8)), ActNormBijection2d(8), Conv1x1(8),
                 #Logit(0.05),
                 ScalarAffineBijection(shift=-0.5),
                 Squeeze2d(),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(4), Conv1x1(4), AffineCouplingBijection(net(4), scale_fn=scale_fn("tanh_exp")),
                 Squeeze2d(),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
                 ActNormBijection2d(16), Conv1x1(16), AffineCouplingBijection(net(16), scale_fn=scale_fn("tanh_exp")),
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
        optimizer.zero_grad()
        #loss = elbo_nats(model, x.to(device))
        loss = elbo_bpd(model, x.to(device))
        #loss = -model.log_prob(x.to(device)).mean()
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
        #loss = iwbo_nats(model, x.to(device), k=10)
        loss = iwbo_bpd(model, x.to(device), k=10)
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
vutils.save_image(samples.cpu().float(), 'mnist_conv_flow.png', nrow=8)
