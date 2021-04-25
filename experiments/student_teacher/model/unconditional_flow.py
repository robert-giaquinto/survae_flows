import torch
import torch.nn as nn

from survae.flows import Flow
from survae.distributions import StandardNormal, StandardUniform
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection, ActNormBijection, Reverse, Augment
from survae.transforms import SimpleAbsSurjection, Logit, SoftplusInverse
from survae.nn.nets import MLP
from survae.nn.layers import ElementwiseParams, LambdaLayer, scale_fn

from model.layers import ShiftBijection, ScaleBijection
from model.layers import ElementAbsSurjection, ShiftBijection, ScaleBijection


class UnconditionalFlow(Flow):

    def __init__(self, num_flows, actnorm, affine, scale_fn_str, hidden_units, activation, range_flow, augment_size, base_dist):

        D = 2 # Number of data dimensions
        
        if base_dist == "uniform":
            classifier = MLP(D, D // 2,
                             hidden_units=hidden_units,
                             activation=activation,
                             out_lambda=lambda x: x.view(-1))
            transforms=[ElementAbsSurjection(classifier=classifier),
                ShiftBijection(shift=torch.tensor([[0.0, 4.0]])),
                ScaleBijection(scale=torch.tensor([[1/4, 1/8]]))]
            base = StandardUniform((D,))

        else:
            A = D + augment_size # Number of augmented data dimensions
            P = 2 if affine else 1 # Number of elementwise parameters

            # initialize flow with either augmentation or Abs surjection
            if augment_size > 0:
                assert augment_size % 2 == 0
                transforms = [Augment(StandardNormal((augment_size,)), x_size=D)]

            else:
                transforms=[SimpleAbsSurjection()]
                if range_flow == 'logit':
                    transforms += [ScaleBijection(scale=torch.tensor([[1/4, 1/4]])), Logit()]
                elif range_flow == 'softplus':
                    transforms += [SoftplusInverse()]
                    
            # apply coupling layer flows
            for _ in range(num_flows):
                net = nn.Sequential(MLP(A//2, P*A//2,
                                        hidden_units=hidden_units,
                                        activation=activation),
                                    ElementwiseParams(P))
                
                if affine:  transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(scale_fn_str)))
                else:       transforms.append(AdditiveCouplingBijection(net))
                if actnorm: transforms.append(ActNormBijection(D))
                transforms.append(Reverse(A))
                
            transforms.pop()
            base = StandardNormal((A,))

        
        super(UnconditionalFlow, self).__init__(base_dist=base, transforms=transforms)
