import torch
import torch.nn as nn

from survae.nn.layers import LambdaLayer
from survae.nn.layers import act_module
from survae.nn.layers import Conv2d, Conv2dZeros, Conv2dResize, GatedConv, GatedConv2d, GatedConvTranspose2d


class ConvNet(nn.Sequential):
    """
    Convolution net useful in coupling layers. Uses 3-1-3 filters. Similar to the Glow Pytorch implementation:
    https://github.com/chaiyujin/glow-pytorch/blob/master/glow/models.py
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


class ConvEncoderNet(nn.Sequential):
    """
    Convolutional encoder for use with a variational autoencoder
    """
    def __init__(self, in_channels, out_channels, mid_channels=[64,128,256], max_pool=True, batch_norm=True, in_lambda=None, out_lambda=None):
        assert isinstance(mid_channels, list) and len(mid_channels) > 0, f"mid_channels={mid_channels}"
        
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        layers.append(_ConvEncoder(in_channels, out_channels, mid_channels=mid_channels, max_pool=max_pool, batch_norm=batch_norm))
        if out_lambda: layers.append(LambdaLayer(out_lambda))
        super(ConvEncoderNet, self).__init__(*layers)


class ConvDecoderNet(nn.Sequential):
    """
    Convolutional decoder for use with a variational autoencoder
    """
    def __init__(self, in_channels, out_shape, mid_channels=[256,128,64], batch_norm=True, in_lambda=None, out_lambda=None):
        assert isinstance(mid_channels, list) and len(mid_channels) > 0, f"mid_channels={mid_channels}"
        
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        layers.append(_ConvDecoder(in_channels, out_shape=out_shape, mid_channels=mid_channels, batch_norm=batch_norm, init_weights=True))
        if out_lambda: layers.append(LambdaLayer(out_lambda))
        super(ConvDecoderNet, self).__init__(*layers)


class ResizeConvNet(nn.Sequential):
    """
    Context convolution net used in the CondAffineCoupling of:
    https://github.com/yolu1055/conditional-glow/blob/master/modules.py

    - Resizes the input to the output size with convolutional layers
    """
    def __init__(self, in_size, out_size, mid_channels, activation='relu', in_lambda=None, out_lambda=None):

        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        
        layers.append(Conv2dZeros(in_size[0], mid_channels))
        if activation is not None: layers.append(act_module(activation, allow_concat=False))
        
        layers.append(Conv2dResize(in_size=(mid_channels, in_size[1], in_size[2]), out_size=out_size))
        if activation is not None: layers.append(act_module(activation, allow_concat=False))
        
        layers.append(Conv2dZeros(out_size[0], out_size[0]))
        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(ResizeConvNet, self).__init__(*layers)


class _Network(nn.Module):
    """
    Simple neural network base class with specific intializations
    """
    def __init__(self):
        super(_Network, self).__init__()

    def forward(self, x):
        pass

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class _ConvDecoder(_Network):
    """
    Convolutional decoder for use with a variational autoencoder
    Only to be called by ConvEncoderNet.

    Uses transposed convolutions to increase the spatial dimensions of the input back to the 
    size of the original input.
    """
    def __init__(self, in_channels, out_shape, mid_channels, batch_norm, init_weights=True):
        super(_ConvDecoder, self).__init__()

        compressed_input = (in_channels + mid_channels[0]) // 4
        feature_layers = [nn.Conv2d(in_channels, compressed_input, kernel_size=1, stride=1)]
        if batch_norm: feature_layers.append(nn.BatchNorm2d(compressed_input))
        feature_layers.append(nn.ReLU(inplace=True))
        for in_size, out_size in zip([compressed_input] + mid_channels[:-1], mid_channels):
            feature_layers.append(nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=0))
            if batch_norm: feature_layers.append(nn.BatchNorm2d(out_size))
            feature_layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*feature_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((out_shape[1], out_shape[2]))
        self.conv1x1 = nn.Conv2d(mid_channels[-1], out_shape[0], kernel_size=1, stride=1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.conv1x1(out)
        return out
        

class Conv2Flat(_Network):
    """
    Encoder portion of variational autoencoder. Does not perform the final splitting and scaling of the mean and variance though
    (see _ConvEncoder for the full module to be used in the VAE).
    This can be used as a convolution network anytime the input needs to be transformed to a flat vector.
    """
    def __init__(self, in_channels, out_channels, mid_channels, max_pool, batch_norm=True, init_weights=True):
        super(Conv2Flat, self).__init__()

        feature_layers = []
        for i, (in_size, out_size) in enumerate(zip([in_channels] + mid_channels[:-1], mid_channels)):
            feature_layers.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=2))
            if batch_norm: feature_layers.append(nn.BatchNorm2d(out_size))
            feature_layers.append(nn.ReLU(inplace=True))
            if max_pool and i < len(mid_channels) - 1:
                feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*feature_layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1x1 = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=1, stride=1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.conv1x1(out)
        return out


class _ConvEncoder(_Network):
    """
    Convolutional encoder for use with a variational autoencoder
    Only to be called by ConvEncoderNet

    Note: out_channels is the size of the latent space. The module will automatically double the encoder's
          output channels and split it into the mean and variance.
    """
    def __init__(self, in_channels, out_channels, mid_channels, max_pool, batch_norm, init_weights=True):
        super(_ConvEncoder, self).__init__()
        self.encode = Conv2Flat(in_channels, out_channels, mid_channels, max_pool, batch_norm, init_weights)
        self.dense = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, stride=1))
        self.var_activation = nn.Sequential(nn.Softplus(), nn.Hardtanh(min_val=0.01, max_val=7.0))

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        hidden = self.encode(x)
        hidden = self.dense(hidden)
        mean, var = hidden.chunk(2, dim=1)
        log_var = self.var_activation(var)
        out = torch.cat((mean, log_var), dim=1)
        return out



