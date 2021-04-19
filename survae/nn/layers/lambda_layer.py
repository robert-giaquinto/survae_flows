import torch
import torch.nn as nn

from survae.utils import checkerboard_split, checkerboard_inverse

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        if lambd is None: lambd = lambda x: x
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Checkerboard(nn.Module):
    def __init__(self, concat_dim=None):
        super(Checkerboard, self).__init__()
        self.concat_dim = concat_dim
        
    def forward(self, x):
        x1, x2 = checkerboard_split(x)
        if self.concat_dim:
            return torch.cat([x1, x2], dim=self.concat_dim)
        else:
            return x1, x2
        
