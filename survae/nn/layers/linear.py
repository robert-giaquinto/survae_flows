import torch
from torch import nn


class LinearNorm(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.normal_(mean=0.0, std=0.1)
        self.bias.data.normal_(mean=0.0, std=0.1)

        
class LinearZeros(nn.Linear):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output
