import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from survae.utils import sum_except_batch
from survae.transforms.bijections.conditional import ConditionalBijection
from survae.nn.layers import LinearZeros
from survae.nn.nets import ConvEncoderNet, Conv2Flat


class _ConditionalActNormBijection(ConditionalBijection):
    """
    conditional act norm for image data

    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
        [2] Structured Output Learning with Conditional Generative Flows
            You Lu & Bert Huang, 2020, https://arxiv.org/abs/1905.13288
    """
    def __init__(self, cond_shape, eps=1e-6):
        super(_ConditionalActNormBijection, self).__init__()
        self.eps = eps

    def forward(self, x, context):
        shift, log_scale = self.compute_stats(context)
        z = (x - shift) * torch.exp(-log_scale)
        ldj = sum_except_batch(-log_scale).expand([x.shape[0]]) * self.ldj_multiplier(x)
        return z, ldj

    def inverse(self, z, context):
        shift, log_scale = self.compute_stats(context)
        return shift + z * torch.exp(log_scale)

    def compute_stats(self, context):
        '''Compute x_mean and x_std'''
        raise NotImplementedError()

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        raise NotImplementedError()


class ConditionalActNormBijection2d(_ConditionalActNormBijection):
    """
    Activation normalization [1] for inputs on the form (B,C,H,W).
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.

    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
        [2] Structured Output Learning with Conditional Generative Flows
            You Lu & Bert Huang, 2020, https://arxiv.org/abs/1905.13288
    """
    def __init__(self, cond_shape, out_channels, eps=1e-6, mid_channels=[32], norm=True):
        super(ConditionalActNormBijection2d, self).__init__(cond_shape, eps)

        self.cond_net = Conv2Flat(in_channels=cond_shape[0],
                                  out_channels=out_channels * 2,
                                  mid_channels=mid_channels,
                                  max_pool=True,
                                  batch_norm=norm)
        
        self.cond_linear = nn.Sequential(
            #LinearZeros(out_channels * 2, out_channels * 2),
            #nn.ReLU(),
            LinearZeros(out_channels * 2, out_channels * 2),
            nn.Tanh())
        
    def compute_stats(self, context):
        """
        Compute shift and log scale parameters
        """
        context = self.cond_net(context)
        context = context.view(context.shape[0], -1)
        context = self.cond_linear(context)
        context = context.view(context.shape[0], -1, 1, 1)
        shift, log_scale = context.chunk(2, dim=1)
        return shift, log_scale
        

    def ldj_multiplier(self, x):
        """
        Multiplier for ldj
        """
        return x.shape[2:4].numel()


    
