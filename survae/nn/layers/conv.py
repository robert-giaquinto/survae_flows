import torch
from torch import nn


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size=[3,3], stride=[1,1]):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight.data.normal_(mean=0.0, std=0.1)

        
class Conv2dResize(nn.Conv2d):
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
