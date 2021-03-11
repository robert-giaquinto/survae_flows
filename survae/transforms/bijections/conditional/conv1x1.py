import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections.conditional import ConditionalBijection
from survae.nn.layers import LinearZeros, LinearNorm, Conv2dResize
from survae.nn.nets import Conv2Flat


class ConditionalConv1x1(ConditionalBijection):
    """
    Conditional Invertible 1x1 Convolution [2].

    Conditioning is designed for image contexts (not class labels).

    The weight matrix is initialized as a random rotation matrix
    but learned via a conditioning network as described in Section 4.1 of [2].

    Args:
        cond_shape (tensor): Shape of context input.
        out_channels (int): Number of channels in the output.
        slogdet_cpu (bool): If True, compute slogdet on cpu (default=True).

    Note:
        torch.slogdet appears to run faster on CPU than on GPU.
        slogdet_cpu is thus set to True by default.

    References:
        [1] Structured Output Learning with Conditional Generative Flows
            You Lou & Bert Huang, 2020, https://arxiv.org/pdf/1905.13288.pdf
        [2] Structured Output Learning with Conditional Generative Flows
            You Lu & Bert Huang, 2020, https://arxiv.org/abs/1905.13288
    """
    def __init__(self, cond_shape, out_channels, slogdet_cpu=True, mid_channels=[32], norm=True):
        super(ConditionalConv1x1, self).__init__()
        self.out_channels = out_channels
        self.slogdet_cpu = slogdet_cpu

        # conditioning network
        self.cond_net = Conv2Flat(in_channels=cond_shape[0],
                                  out_channels=out_channels * out_channels,
                                  mid_channels=mid_channels,
                                  max_pool=True,
                                  batch_norm=norm)
        self.cond_linear = nn.Sequential(
            #LinearZeros(out_channels * out_channels, out_channels * out_channels),
            #nn.ReLU(),
            LinearNorm(out_channels * out_channels, out_channels * out_channels),
            nn.Tanh()
        )

    def get_weight(self, x, context, reverse):
        x_channels = x.size(1)
        B, C, H, W = context.size()
        context = self.cond_net(context)
        context = context.view(B, -1)
        context = self.cond_linear(context)
        weight = context.view(B, self.out_channels, self.out_channels)

        if reverse == False:
            dimensions = x.size(2) * x.size(3)
            if self.slogdet_cpu:
                logdet = torch.slogdet(weight.to('cpu'))[1] * dimensions
            else:
                logdet = torch.slogdet(weight)[1] * dimensions

            weight = weight.view(B, self.out_channels, self.out_channels, 1, 1)
            return weight, logdet.to(weight.device)
        else:
            weight = torch.inverse(weight.double()).float().view(B, self.out_channels, self.out_channels, 1, 1)
            return weight

    def forward(self, x, context):
        weight, logdet = self.get_weight(x, context, reverse=False)
        B, C, H, W = x.size()
        x = x.view(1, B*C, H, W)
        B_k, C_i_k, C_o_k, H_k, W_k = weight.size()
        assert B == B_k and C == C_i_k and C == C_o_k, "The input and kernel dimensions are different"
        weight = weight.view(B_k * C_i_k, C_o_k, H_k, W_k)

        z = F.conv2d(x, weight, groups=B)
        z = z.view(B, C, H, W)

        return z, logdet

    def inverse(self, x, context):
        weight = self.get_weight(x, context, reverse=True)
        B, C, H, W = x.size()
        x = x.view(1, B*C, H, W)
        B_k, C_i_k, C_o_k, H_k, W_k = weight.size()
        assert B == B_k and C == C_i_k and C == C_o_k, "The input and kernel dimensions are different"
        weight = weight.view(B_k * C_i_k, C_o_k, H_k, W_k)

        z = F.conv2d(x, weight, groups=B)
        z = z.view(B, C, H, W)

        return z

