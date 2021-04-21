import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.distributions import Distribution
from survae.utils import sum_except_batch

try:
    from torch.linalg import norm
except:
    from torch import norm


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples, temperature=None):
        temperature = 1.0 if temperature is None else temperature
        z = torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype) * temperature
        #z = torch.normal(mean=0, std=temperature, size=(num_samples, *self.shape), device=self.buffer.device, dtype=self.buffer.dtype)            
        return z

    def interpolate(self, num_samples, z1=None, z2=None):
        if z1 is None and z2 is None:
            #z1 = torch.randn(1, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype) * 0.01
            #z2 = (torch.round(torch.rand(1, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)) * 2.0) - 1.0

            # select z2 as point on tail with ~5% probability
            z2 = torch.randn(1, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)
            z2 = 1.4 * z2 / norm(z2)
            z1 = z2 * -1  # opposite tail
        elif z2 is None:
            # find unit vector throuh z1, and scale it to a point with ~5% probability
            z2 = 1.4 * z1 / norm(z1)
            z1 = z2 * -1  # opposite tail
        else:
            assert z1.shape == z2.shape

        return torch.cat([w * z2 + (1.0 - w) * z1 for w in np.linspace(0, 1, num_samples)], dim=0)


class DiagonalNormal(Distribution):
    """A multivariate Normal with diagonal covariance."""

    def __init__(self, shape):
        super(DiagonalNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.loc = nn.Parameter(torch.zeros(shape))
        self.log_scale = nn.Parameter(torch.zeros(shape))

    def log_prob(self, x):
        log_base =  - 0.5 * math.log(2 * math.pi) - self.log_scale
        log_inner = - 0.5 * torch.exp(-2 * self.log_scale) * ((x - self.loc) ** 2)
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples, temperature=None):
        eps = torch.randn(num_samples, *self.shape, device=self.loc.device, dtype=self.loc.dtype)
        z = self.loc + self.log_scale.exp() * eps
        if temperature is None:
            return z
        else:
            return z * temperature

    def interpolate(self, num_samples, z1=None, z2=None):
        if z1 is None or z2 is None:
            #z1 = torch.randn(1, *self.shape, device=self.loc.device, dtype=self.loc.dtype) * 0.01 + self.loc
            #eps = (torch.round(torch.rand(1, *self.shape, device=self.loc.device, dtype=self.loc.dtype)) * 2.0) - 1.0

            # select z2 as point on tail with ~5% probability
            eps = torch.randn(1, *self.shape, device=self.loc.device, dtype=self.loc.dtype)
            z2 = 1.4 * eps / norm(eps)
            z2 = self.loc + self.log_scale.exp() * eps
            z1 = z2 * -1  # opposite tail
        elif z2 is None:
            # rename points so that z1 still represents point near the origin
            z2 = z1
            z1 = z2 * -1.0  # opposite tail
        else:
            assert z1.shape == z2.shape

        return torch.cat([w * z2 + (1.0 - w) * z1 for w in np.linspace(0, 1, num_samples)], dim=0)
            

class ConvNormal2d(DiagonalNormal):
    def __init__(self, shape):
        super(DiagonalNormal, self).__init__()
        assert len(shape) == 3, f"len(shape) must be 3, not {len(shape)}"
        self.shape = torch.Size(shape)
        self.loc = nn.Parameter(torch.zeros(1, shape[0], 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, shape[0], 1, 1))
