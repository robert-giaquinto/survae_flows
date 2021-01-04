import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections.conditional import ConditionalBijection
from survae.nn.layers import LinearZeros, LinearNorm


class ConditionalConv1x1(ConditionalBijection):
    """
    Conditional Invertible 1x1 Convolution [2].

    Conditioning is designed for image contexts (not class labels).

    The weight matrix is initialized as a random rotation matrix
    but learned via a conditioning network as described in Section 4.1 of [2].

    Args:
        cond_shape (tensor): Shape of context input.
        num_channels (int): Number of channels in the input and output.
        orthogonal_init (bool): If True, initialize weights to be a random orthogonal matrix (default=True).
        slogdet_cpu (bool): If True, compute slogdet on cpu (default=True).

    Note:
        torch.slogdet appears to run faster on CPU than on GPU.
        slogdet_cpu is thus set to True by default.

    References:
        [1] Structured Output Learning with Conditional Generative Flows
            You Lou & Bert Huang, 2020, https://arxiv.org/pdf/1905.13288.pdf

    TODO: Currently only set up to for MNIST, the input and output sizes need to be more dynamic.
    """
    def __init__(self, cond_shape, num_channels, orthogonal_init=True, slogdet_cpu=True):
        super(ConditionalConv1x1, self).__init__()
        self.num_channels = num_channels
        self.slogdet_cpu = slogdet_cpu

        #cond_channels = 256
        cond_channels = 64
        cond_size = 128
        C_cond, H_cond, W_cond = cond_shape

        # conditioning network
        self.cond_net = nn.Sequential(
            Conv2dResize(in_size=[C_cond, H_cond, W_cond], out_size=[cond_channels, H_cond//2, W_cond//2]),
            nn.ReLU(),
            Conv2dResize(in_size=[cond_channels, H_cond//2, W_cond//2], out_size=[cond_channels * 2, H_cond//7, W_cond//7]),
            # Conv2dResize(in_size=[cond_channels, H_cond//2, W_cond//2], out_size=[cond_channels * 2, H_cond//4, W_cond//4]),
            # nn.ReLU(),
            # Conv2dResize(in_size=[cond_channels * 2, H_cond//4, W_cond//4], out_size=[cond_channels * 4, H_cond//8, W_cond//8]),
            nn.ReLU()
        )

        self.cond_linear = nn.Sequential(
            # LinearZeros(cond_channels * 4 * H_cond * W_cond // (8*8), cond_size),
            LinearZeros(cond_channels * 2 * H_cond * W_cond // (7*7), cond_size),
            nn.ReLU(),
            # LinearZeros(cond_size, cond_size),
            # nn.ReLU(),
            LinearNorm(cond_size, num_channels * num_channels),
            nn.Tanh()
        )

    def _conv(self, weight, v):
        return F.conv2d(v, weight.unsqueeze(-1).unsqueeze(-1))

    def get_weight(self, x, context, reverse):
        x_channels = x.size(1)
        B, C, H, W = context.size()
        context = self.cond_net(context)
        context = context.view(B, -1)
        context = self.cond_linear(context)
        weight = context.view(B, self.num_channels, self.num_channels)

        dimensions = x.size(2) * x.size(3)
        if self.slogdet_cpu:
            logdet = torch.slogdet(weight.to('cpu'))[1] * dimensions
        else:
             logdet = torch.slogdet(weight)[1] * dimensions

        if reverse == False:
            weight = weight.view(B, self.num_channels, self.num_channels, 1, 1)
        else:
            weight = torch.inverse(weight.double()).float().view(B, self.num_channels, self.num_channels, 1, 1)

        return weight, logdet.to(weight.device)

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
        weight, logdet = self.get_weight(x, context, reverse=True)
        B, C, H, W = x.size()
        x = x.view(1, B*C, H, W)
        B_k, C_i_k, C_o_k, H_k, W_k = weight.size()
        assert B == B_k and C == C_i_k and C == C_o_k, "The input and kernel dimensions are different"
        weight = weight.view(B_k * C_i_k, C_o_k, H_k, W_k)

        z = F.conv2d(x, weight, groups=B)
        z = z.view(B, C, H, W)

        return z, logdet


    
class Conv2dZeros(nn.Conv2d):
    """
    Convolutional 2d layer specific for the conditional 1x1 invertible convolutioon
    """
    def __init__(self, in_channel, out_channel, kernel_size=[3,3], stride=[1,1]):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight.data.normal_(mean=0.0, std=0.1)

        
class Conv2dResize(nn.Conv2d):
    """
    Convolutional 2d layer specific for the conditional 1x1 invertible convolutioon
    """
    def __init__(self, in_size, out_size):
        stride = [in_size[1]//out_size[1], in_size[2]//out_size[2]]
        kernel_size = Conv2dResize.compute_kernel_size(in_size, out_size, stride)
        super().__init__(in_channels=in_size[0], out_channels=out_size[0], kernel_size=kernel_size, stride=stride)
        self.weight.data.zero_()

    @staticmethod
    def compute_kernel_size(in_size, out_size, stride):
        k0 = in_size[1] - (out_size[1] - 1) * stride[0]
        k1 = in_size[2] - (out_size[2] - 1) * stride[1]
        return[k0,k1]
