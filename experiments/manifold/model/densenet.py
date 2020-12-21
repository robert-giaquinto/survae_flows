import torch
import torch.nn as nn

from survae.nn.nets import MLP, DenseNet
from survae.nn.layers import ElementwiseParams2d


def densenet(channels,
	num_blocks=1,
	mid_channels=64,
	depth=2,
	growth=16,
	dropout=0.0,
	gated_conv=True,
	zero_init=True):

    return nn.Sequential(DenseNet(in_channels=channels//2,
                                  out_channels=channels,
                                  num_blocks=num_blocks,
                                  mid_channels=mid_channels,
                                  depth=depth,
                                  growth=growth,
                                  dropout=dropout,
                                  gated_conv=gated_conv,
                                  zero_init=zero_init),
                         ElementwiseParams2d(2))
