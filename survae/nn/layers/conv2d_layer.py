import torch
import torch.nn as nn
import torch.nn.functional as F

from survae.nn.layers import act_module

        
class GatedConv(nn.Module):
    """
    Gated Convolution Block
    Originally used by PixelCNN++ (https://arxiv.org/pdf/1701.05517).
    Args:
        channels (int): Number of channels in hidden activations.
        context_channels (int): (Optional) Number of channels in optional contextual side input.
        dropout (float): Dropout probability.
    """
    def __init__(self, channels, context_channels=None, dropout=0.0, weight_norm=True):
        super(GatedConv, self).__init__()
        self.activation = act_module('concat_elu', allow_concat=True)
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(2 * channels, channels, kernel_size=3, padding=1, weight_norm=weight_norm)
        self.drop = nn.Dropout2d(dropout)
        self.gate = Conv2d(2 * channels, 2 * channels, kernel_size=1, padding=0, weight_norm=weight_norm)
        if context_channels is not None:
            self.context_conv = Conv2d(2 * context_channels, channels, kernel_size=1, padding=0, weight_norm=weight_norm)
        else:
            self.context_conv = None

    def forward(self, x, context=None):
        x = self.activation(x)
        x = self.conv(x)
        if context is not None:
            context = self.activation(context)
            x = x + self.aux_conv(context)
        x = self.activation(x)
        x = self.drop(x)
        x = self.gate(x)
        h, g = x.chunk(2, dim=1)
        x = h * torch.sigmoid(g)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same", weight_std=0.05, weight_norm=True):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        if weight_norm:
            self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv2dZeros(nn.Module):
    """
    2D convolutional layer with zero initialization similar to the Glow Pytorch implementation:
    https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", logscale_factor=3):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, sample):
        output = self.conv(sample)
        return output * torch.exp(self.logs * self.logscale_factor)

        
class Conv2dResize(nn.Conv2d):
    """
    Convolutional 2d layer specific for the conditional 1x1 invertible convolution and coupling layers
    - Allows the conditional/context inputs to be resized to the same size and other (primary) flow inputs

    https://github.com/yolu1055/conditional-glow/blob/master/modules.py
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
    

class GatedConv2d(nn.Module):
    """
    Simple gated convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same", activation=None, weight_norm=True):
        super(GatedConv2d, self).__init__()
        
        self.activation = activation
        self.conv = Conv2d(in_channels, out_channels*2, kernel_size=kernel_size, padding=padding, stride=stride, weight_norm=weight_norm)
        
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)

        h, g = x.chunk(2, dim=1)
        hg = h * torch.sigmoid(g)
        return hg


class GatedConvTranspose2d(nn.Module):
    """
    Simple gated de-convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding="same", out_padding=0, activation=None, weight_norm=True):
        super(GatedConvTranspose2d, self).__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.activation = activation
        if weight_norm:
            self.conv = nn.utils.weight_norm(
                nn.ConvTranspose2d(in_channels, out_channels * 2, kernel_size, stride, padding, out_padding, dilation=1))
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels * 2, kernel_size, stride, padding, out_padding, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)

        h, g = x.chunk(2, dim=1)
        hg = h * torch.sigmoid(g)
        return hg
    

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size),\
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


