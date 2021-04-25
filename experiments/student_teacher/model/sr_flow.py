import torch
import torch.nn as nn

from survae.flows import Flow, ConditionalFlow
from survae.distributions import StandardNormal
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection, ActNormBijection, Reverse, Augment
from survae.transforms import ConditionalAdditiveCouplingBijection, ConditionalAffineCouplingBijection
from survae.transforms import SimpleAbsSurjection, Logit, SoftplusInverse
from survae.nn.nets import MLP
from survae.nn.layers import ElementwiseParams, LambdaLayer, scale_fn

from model.layers import ShiftBijection, ScaleBijection


class ContextInit(nn.Module):
    def __init__(self, in_size, out_size):
        super(ContextInit, self).__init__()
        self.dense = nn.Sequential(nn.Linear(in_size, out_size), nn.ReLU())
        
    def forward(self, context):
        return self.dense(context)


class SRFlow(ConditionalFlow):

    def __init__(self, num_flows, actnorm, affine, scale_fn_str, hidden_units, activation, range_flow, augment_size, base_dist):
        
        D = 2 # Number of data dimensions
        A = D + augment_size # Number of augmented data dimensions
        P = 2 if affine else 1 # Number of elementwise parameters

        # initialize context. Only upsample context in ContextInit if latent shape doesn't change during the flow.
        context_size = D
        context_init = ContextInit(context_size, context_size)

        # initialize flow with either augmentation or Abs surjection
        if augment_size > 0:
            assert augment_size % 2 == 0
            transforms = [Augment(StandardNormal((augment_size,)), x_size=D)]

        else:
            transforms = []
            # transforms=[SimpleAbsSurjection()]
            # if range_flow == 'logit':
            #     transforms += [ScaleBijection(scale=torch.tensor([[1/4, 1/4]])), Logit()]
            # elif range_flow == 'softplus':
            #     transforms += [SoftplusInverse()]
                    
        # apply coupling layer flows
        for _ in range(num_flows):
            net = nn.Sequential(MLP(A // 2 + context_size,
                                    P * A // 2,
                                    hidden_units=hidden_units,
                                    activation=activation),
                                ElementwiseParams(P))
            if affine:  transforms.append(ConditionalAffineCouplingBijection(net, scale_fn=scale_fn(scale_fn_str)))
            else:       transforms.append(ConditionalAdditiveCouplingBijection(net))
            if actnorm: transforms.append(ActNormBijection(D))
            transforms.append(Reverse(A))
            
        transforms.pop()

        if base_dist == "uniform":
            base = StandardUniform((A,))
        else:
            base = StandardNormal((A,))
        
        super(SRFlow, self).__init__(base_dist=base, transforms=transforms, context_init=context_init)
