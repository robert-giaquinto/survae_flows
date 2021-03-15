import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform


class CompressiveFlow(Distribution):
    """
    Base class for non-dimension preserving / Compressive Normalizing Flow.

    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.

    The last transform given is assumed to be the transform that changes
    the flows dimensionality.

    base_dist must be a an iterable of Distributions, of length 2:
    the base_dist[0] matches the size of the output from the bijective flows,
    and can be set to None of the 
    Next, base_dist[1] matches the shape of the change in dimension flows, etc.
    """

    def __init__(self, base_dist, transforms):
        super(CompressiveFlow, self).__init__()
        assert isinstance(base_dist, Iterable)
        assert all(base is None or isinstance(base, Distribution) for base in base_dist)

        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)

        self.base_dist = nn.ModuleList(base_dist)
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x, beta=1.0):
        """
        Calculate log probability of x.
        Args:
            x    = Input data
            beta = Beta annealing scalar, controlling the weight of the NDP portion of the flow.
                   Beta can be annealed from 0 to 1 over the first few epochs, however this is
                   only applicable when the bijective portion of the flow is regularized to
                   have a Gaussian output.

        TODO: pass transforms as a list of lists, where the length corresponds to each base_dist
             the sublists include all transforms for a particular bijective piece of the flow with
             the change of dimenion flow being the final element.
        """
        log_prob = torch.zeros(x.shape[0], device=x.device)

        # Dimension preserving flows
        for transform in self.transforms[:-1]:
            x, ldj = transform(x)
            log_prob += ldj
        
        # can impose pre-ndp transformation to be distributed like base_dist[0]
        if self.base_dist[0] is not None and self.training:
            log_prob += self.base_dist[0].log_prob(x)
        else:
            # using a single base distribution
            assert beta == 1.0, f'Beta annealing for NDP flows is only applicable when using 2 base distributions.'

        # Non-dimension preserving flow
        x, ndp_ldj = self.transforms[-1](x)
        log_pz = self.base_dist[-1].log_prob(x) # if self.base_dist[1] is not None else 0.0

        log_prob += beta * (log_pz + ndp_ldj)
        return log_prob

    def sample(self, num_samples):
        x = self.base_dist[-1].sample(num_samples)
        for transform in reversed(self.transforms):
            x = transform.inverse(x)

        return x

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")
