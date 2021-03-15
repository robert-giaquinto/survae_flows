import torch
import torch.nn as nn


class IdentityTransform(nn.Module):
    """
    Applies no transformation, but makes the printed architecture of the model look nice
    """
    def __init__(self):
        super(IdentityTransform, self).__init__()

    def forward(self, x):
        return x



class UpsamplerNet(nn.Module):
    """
    Uses transposed convolutions to increase the spatial dimensions of the input back to the 
    size of the original input.
    """
    def __init__(self, in_channels, out_shape, mid_channels, batch_norm=True, init_weights=True):
        super(UpsamplerNet, self).__init__()
        assert isinstance(mid_channels, list) and len(mid_channels) > 0, f"mid_channels={mid_channels}"

        feature_layers = []
        for in_size, out_size in zip([in_channels] + mid_channels[:-1], mid_channels):
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


class ContextUpsampler(nn.Module):
    """
    Wrapper module around a network that will only apply the 
    forward pass if called in specified forward or invere direction
    """
    def __init__(self, context_net, direction):
        super(ContextUpsampler, self).__init__()

        self.direction = direction
        assert direction in ['forward', 'inverse']
        if direction == 'forward':
            self.forward_only = context_net
            self.inverse_only = IdentityTransform()
        else:
            self.forward_only = IdentityTransform()
            self.inverse_only = context_net


    def forward(self, context):
        # if self.forward_transformation:
        #     return self.forward_transformation(context)
        # else:
        #     return context
        return self.forward_only(context)

    def inverse(self, context):
        # if self.inverse_transformation:
        #     return self.inverse_transformation(context)
        # else:
        #     return context
        return self.inverse_only(context)
