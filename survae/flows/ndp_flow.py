import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform


class NDPFlow(Distribution):
    """
    Base class for Non-Dimension Preserving Flow.

    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.

    The last transform given is assumed to be the transform that changes
    the flows dimensionality.

    base_dist must be a an iterable of Distributions, generally of length 2 or more:
    the base_dist[0] is of same dimension as data, and base_dist[1] changes
    dimension, etc.

    TODO: base_dist should match length of transforms. each
          transform will corresponding to a sequence of flow steps
    """

    def __init__(self, base_dist, transforms):
        super(NDPFlow, self).__init__()
        assert isinstance(base_dist, Iterable)
        assert all(base is None or isinstance(base, Distribution) for base in base_dist)

        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)

        self.base_dist = nn.ModuleList(base_dist)
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        # Dimension preserving flows
        for transform in self.transforms[:-1]:
            x, ldj = transform(x)
            log_prob += ldj
        log_prob += self.base_dist[0].log_prob(x)

        # Non-dimension preserving flow
        x, ldj = self.transforms[-1](x)
        log_prob += ldj
        if self.base_dist[1] is not None:
            log_prob += self.base_dist[1].log_prob(x)
        return log_prob

    def sample(self, num_samples):
        # map low dimensional noise to higher dimensions
        z = self.base_dist[-1].sample(num_samples)
        x = self.transforms[-1].inverse(z)
        for transform in reversed(self.transforms[:-1]):
            x = transform.inverse(x)
        return x

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")
