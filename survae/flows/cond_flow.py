import torch
from torch import nn
from collections.abc import Iterable
from survae.utils import context_size
from survae.distributions import Distribution, ConditionalDistribution
from survae.transforms import Transform, ConditionalTransform
from survae.nn.nets import ContextUpsampler


class ConditionalFlow(ConditionalDistribution):
    """
    Base class for ConditionalFlow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, base_dist, transforms, context_init=None):
        super(ConditionalFlow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) or isinstance(transform, ContextUpsampler) for transform in transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)
        self.context_init = context_init
        self.transforms = nn.ModuleList(transforms)
        self.base_dist = base_dist

    def log_prob(self, x, context):
        if self.context_init: context = self.context_init(context)
        encoded_context = context
        
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                x, ldj = transform(x, encoded_context)
                log_prob += ldj
            elif isinstance(transform, Transform):
                x, ldj = transform(x)
                log_prob += ldj
            elif isinstance(transform, ContextUpsampler):
                if transform.direction == "forward": encoded_context = transform(context)
            
        if isinstance(self.base_dist, ConditionalDistribution):
            log_prob += self.base_dist.log_prob(x, encoded_context)
        else:
            log_prob += self.base_dist.log_prob(x)
            
        return log_prob

    def sample(self, context, temperature):
        if self.context_init: context = self.context_init(context)
        encoded_context = context

        if isinstance(self.base_dist, ConditionalDistribution):
            z = self.base_dist.sample(encoded_context)
        else:
            z = self.base_dist.sample(context_size(encoded_context), temperature=temperature)

        for transform in reversed(self.transforms):            
            if isinstance(transform, ConditionalTransform):
                z = transform.inverse(z, encoded_context)
            elif isinstance(transform, Transform):
                z = transform.inverse(z)
            elif isinstance(transform, ContextUpsampler):
                if transform.direction == "inverse": encoded_context = transform.inverse(context)
            
        return z

    def sample_with_log_prob(self, context):
        raise RuntimeError("ConditionalFlow does not support sample_with_log_prob, see ConditionalInverseFlow instead.")
