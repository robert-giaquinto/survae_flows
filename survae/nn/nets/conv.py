import torch
import torch.nn as nn

from survae.nn.layers import LambdaLayer
from survae.nn.layers import act_module
from survae.nn.layers import Conv2d, Conv2dZeros, GatedConv


class ConvNet(nn.Sequential):
    """
    Convolution net useful in coupling layers. Uses 3-1-3 filters.
    """
    def __init__(self, in_channels, out_channels, mid_channels, num_layers=1, activation='relu', weight_norm=True, in_lambda=None, out_lambda=None):

        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        layers += [Conv2d(in_channels, mid_channels, weight_norm=weight_norm)]
        for i in range(num_layers):
            if activation is not None: layers.append(act_module(activation, allow_concat=True))
            layers.append(Conv2d(mid_channels, mid_channels, kernel_size=(1, 1), weight_norm=weight_norm))

        if activation is not None: layers.append(act_module(activation, allow_concat=True))
        layers.append(Conv2dZeros(mid_channels, out_channels))
        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(ConvNet, self).__init__(*layers)


class GatedConvNet(nn.Sequential):
    """
    Gated convolutional neural network layer.
    """
    def __init__(self, channels, context_channels=None, num_blocks=1, dropout=0.0, in_lambda=None, out_lambda=None):
        assert num_blocks > 0
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        layers += [GatedConv(channels, context_channels=context_channels, dropout=dropout) for i in range(num_blocks)]
        if out_lambda: layers.append(LambdaLayer(out_lambda))
        super(GatedConvNet, self).__init__(*layers)





