import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform
from survae.flows import Flow


class BoostedFlow(Distribution):
    """
    Base class for gradient boosted normalizing flows.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, flows, args):
        super(BoostedFlow, self).__init__()

        assert isinstance(flows, Iterable)
        assert len(flows) == args.boosted_components, f"len(flows)={len(flows)} and args.boosted_components={args.boosted_components}"
        self.flows = nn.ModuleList(flows)

        self.mixture_type = "multiplicative"
        self.num_components = args.boosted_components
        self.component = 0 if args.pretrained_model is None else 1  # initial component to train

        if args.device == "cuda":
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor
            
        # Initialize weights rho
        if args.rho_init == "decreasing":
            # each component is given half the weight of the previous one
            self.register_buffer('rho', torch.clamp(
                1.0 / torch.pow(2.0, self.FloatTensor(self.num_components).fill_(0.0) + \
                                torch.arange(self.num_components * 1.0, device=args.device)), min=0.05).to(args.device))
        else:
            # model weight rho are uniformly weighted
            self.register_buffer('rho', self.FloatTensor(self.num_components).fill_(1.0 / self.num_components))

    def log_prob(self, x):
        """
        log probability calculation for a new component in gradient boosted normalizing flows
        - uses multiplicative mixture model
        - computes loss for new component by resampling x based on fixed component's loss. This tends 
          to be more effective than using the reweighted loss g_ll / G_ll

        TODO: cache results for fixed_nll from previous fixed components
        """
        if self.component == 0:
            return self.flows[0].log_prob(x)

        if self.mixture_type == "multiplicative":
            mixture_log_prob = self.mult_mixture_log_prob
        elif self.mixture_type == "additive":
            mixture_log_prob = self.add_mixture_log_prob

        if self.training:
            # 1. compute weight for each observation according to fixed components
            fixed_nll = -mixture_log_prob(x, sum_over_fixed=True)

            # 2. sample x with replacement
            x_weights = self.normalize(fixed_nll, softmax=True, max_wt=0.1)
            x_resampled = x[torch.multinomial(x_weights, x.size(0), replacement=True)]

            # 3. compute new component's log_prob for the resampled x
            log_prob = self.flows[self.component].log_prob(x_resampled)

        else:
            log_prob = mixture_log_prob(x, sum_over_fixed=False)
            
        return log_prob

    def mult_mixture_log_prob(self, x, sum_over_fixed=False):
        if self.component == 0:
            return self.flows[0].log_prob(x)

        num_components = self.component # sum over first c-1 components
        if not sum_over_fixed:
            num_components += 1 # sum over the first c components

        log_prob = torch.zeros(x.shape[0], device=x.device)
        for c in range(num_components):
            rho = self.normalize(self.rho[:c+1])
            log_prob += rho[c] * self.flows[c].log_prob(x)

        return log_prob

    def add_mixture_log_prob(self, x, sum_over_fixed=False):
        """
        log probability for the full ensemble of gradient boosted components
        """
        if self.component == 0:
            return self.flows[0].log_prob(x)

        num_components = self.component # sum over first c-1 components
        if not sum_over_fixed:
            num_components += 1 # sum over the first c components

        log_prob = torch.zeros(x.shape[0], device=x.device)
        for c in range(num_components):
            log_prob_c = self.flows[c].log_prob(x)
            if c == 0:
                log_prob = log_prob_c
            else:
                rho = self.normalize(self.rho[:c+1])
                last_prob = torch.log(1 - rho[c]) + log_prob
                curr_prob = torch.log(rho[c]) + log_prob_c
                unnormalized = torch.cat([last_prob.view(x.size(0), 1), curr_prob.view(x.size(0), 1)], dim=1)
                log_prob = torch.logsumexp(unnormalized, dim=1)

        return log_prob

    def approximate_mixture_log_prob(self, x):
        """
        Approximates the true log-probability for the multiplicative ensemble by sampling a single component
        for each sample x and evaluating the log probability for some component. Each component
        has an assigned weight, and will be sampled according to that weight.

        Can average log_probs over a number of random samples

        This only an alterantive (possibly more efficient) way to evaluate the model, not for training
        """
        if self.component == 0:
            return self.flows[0].log_prob(x)

        num_samples = self.num_components
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for i in range(num_samples):
            c = self._sample_component("1:c")
            log_prob += self.flows[c].log_prob(x)

        return log_prob / (1.0 * num_samples)
        
    def sample(self, num_samples, component="1:c"):
        c = self._sample_component(component)
        z = self.flows[c].sample(num_samples)
        return z

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")

    def normalize(self, x, softmax=False, max_wt=None):
        if len(x) == 1:
            return torch.ones_like(x)
        
        if softmax:
            wts = torch.exp(x - torch.logsumexp(x, dim=0))
            if max_wt is not None:
                wts = torch.clamp(wts, min=max_wt/x.shape[0], max=max_wt)
                wts = wts / torch.sum(wts)
        else:
            wts = x / torch.sum(x)

        return wts

    def update_rho(self, data_loader):
        """
        Not needed for multiplicative flow
        """
        pass

    def increment_component(self):
        self.component = min(self.component + 1, self.num_components)
            
    def _sample_component(self, sampling_components):
        """
        Given the argument sampling_components (such as "1:c", "1:c-1", or "-c"), sample a component id from the possible
        components specified by the keyword.

        "1:c":   sample from any of the first c components 
        "1:c-1": sample from any of the first c-1 components 
        "-c":    sample from any component except the c-th component (used during a second pass when fine-tuning components)

        Returns the integer id of the sampled component
        """
        if isinstance(sampling_components, int):
            assert sampling_components < self.num_components
            return sampling_components
        
        if sampling_components == "c":
            # sample from new component
            j = min(self.component, self.num_components - 1)
            
        elif sampling_components in ["1:c", "1:c-1"]:
            # sample from either the first 1:c-1 (fixed) or 1:c (fixed + new = all) components
            if sampling_components == "1:c-1":
                num_components = self.component
            elif sampling_components == "1:c":
                num_components = min(self.num_components, self.component + 1)
                
            num_components = min(max(num_components, 1), self.num_components)
            rho_simplex = self.rho[0:num_components] / torch.sum(self.rho[0:num_components])
            j = torch.multinomial(rho_simplex, 1, replacement=True).item()
                
        elif sampling_components == "-c":
            rho_simplex = self.rho.clone().detach()
            rho_simplex[self.component] = 0.0
            rho_simplex = rho_simplex / rho_simplex.sum()
            j = torch.multinomial(rho_simplex, 1, replacement=True).item()

        else:
            raise ValueError("z_k can only be sampled from ['c', '1:c-1', '1:c', '-c'] (corresponding to 'new', 'fixed', or new+fixed components)")

        return j

