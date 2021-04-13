import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform
from survae.flows import BoostedFlow, Flow


class ConditionalBoostedFlow(BoostedFlow):
    """
    Base class for conditional gradient boosted normalizing flows.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, flows, args):
        super(ConditionalBoostedFlow, self).__init__(flows, args)
        
    def log_prob(self, x, context):
        """
        log probability calculation for a new component in gradient boosted normalizing flows
        - uses multiplicative mixture model
        - computes loss for new component by resampling x based on fixed component's loss. This tends 
          to be more effective than using the reweighted loss g_ll / G_ll

        TODO: cache results for fixed_nll from previous fixed components
        """
        if self.component == 0:
            return self.flows[0].log_prob(x, context)

        if self.mixture_type == "multiplicative":
            mixture_log_prob = self.mult_mixture_log_prob
        elif self.mixture_type == "additive":
            mixture_log_prob = self.add_mixture_log_prob

        if self.training:
            # 1. compute weight for each observation according to fixed components
            fixed_nll = -mixture_log_prob(x, context, sum_over_fixed=True)

            # 2. sample x with replacement
            x_weights = self.normalize(fixed_nll, softmax=True, max_wt=0.1)
            resampled_ids = torch.multinomial(x_weights, x.size(0), replacement=True)
            x_resampled = x[resampled_ids]
            context_resampled = context[resampled_ids]

            # 3. compute new component's log_prob for the resampled x
            log_prob = self.flows[self.component].log_prob(x_resampled, context_resampled)

        else:
            log_prob = mixture_log_prob(x, context, sum_over_fixed=False)
            
        return log_prob

    def mult_mixture_log_prob(self, x, context, sum_over_fixed=False):
        """
        log probability for the full ensemble of gradient boosted components
        """        
        if self.component == 0:
            return self.flows[0].log_prob(x, context)

        num_components = self.component # sum over first c-1 components
        if not sum_over_fixed:
            num_components += 1 # sum over the first c components

        log_prob = torch.zeros(x.shape[0], device=x.device)
        for c in range(num_components):
            rho = self.normalize(self.rho[:c+1])
            log_prob += rho[c] * self.flows[c].log_prob(x, context)

        return log_prob

    def approximate_mixture_log_prob(self, x, context):
        """
        Approximates the true log-probability for the ensemble by sampling a single component
        for each sample x and evaluating the log probability for some component. Each component
        has an assigned weight, and will be sampled according to that weight.

        Can average log_probs over a number of random samples
        """
        if self.component == 0:
            return self.flows[0].log_prob(x, context)

        num_samples = self.num_components
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for i in range(num_samples):
            c = self._sample_component("1:c")
            log_prob += self.flows[c].log_prob(x, context)

        return log_prob / (1.0 * num_samples)

    def add_mixture_log_prob(self, x, context):
        """
        log probability for the full ensemble of gradient boosted components
        """        
        if self.component == 0:
            return self.flows[0].log_prob(x, context)

        num_components = self.component # sum over first c-1 components
        if not sum_over_fixed:
            num_components += 1 # sum over the first c components

        log_prob = torch.zeros(x.shape[0], device=x.device)
        for c in range(num_components):
            log_prob_c = self.flows[c].log_prob(x, context)
            if c == 0:
                log_prob = log_prob_c
            else:
                rho = self.normalize(self.rho[:c+1])
                last_prob = torch.log(1 - rho[c]) + log_prob
                curr_prob = torch.log(rho[c]) + log_prob_c
                unnormalized = torch.cat([last_prob.view(x.size(0), 1), curr_prob.view(x.size(0), 1)], dim=1)
                log_prob = torch.logsumexp(unnormalized, dim=1)

        return log_prob

    def interpolate(self, num_samples, context, component="1:c"):
        c = self._sample_component(component)
        z = self.flows[c].interpolate(num_samples, context)
        return z

        
