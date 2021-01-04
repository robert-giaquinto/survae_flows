import torch
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional.mixtures import gaussian_mixture_transform, logistic_mixture_transform, censored_logistic_mixture_transform
from survae.transforms.bijections.functional.mixtures import get_mixture_params, get_flowpp_params
from survae.transforms.bijections.conditional.coupling import ConditionalCouplingBijection


class ConditionalGaussianMixtureCouplingBijection(ConditionalCouplingBijection):

    def __init__(self, coupling_net, num_mixtures, context_net=None, split_dim=1, num_condition=None):
        super(ConditionalGaussianMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, context_net=context_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = gaussian_mixture_transform(inputs=inputs,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)


class ConditionalLogisticMixtureCouplingBijection(ConditionalCouplingBijection):

    def __init__(self, coupling_net, num_mixtures, context_net=None, split_dim=1, num_condition=None):
        super(ConditionalLogisticMixtureCouplingBijection, self).__init__(
            coupling_net=coupling_net, context_net=context_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)


        z, ldj_elementwise = logistic_mixture_transform(inputs=inputs,
                                                        logit_weights=logit_weights,
                                                        means=means,
                                                        log_scales=log_scales,
                                                        eps=self.eps,
                                                        max_iters=self.max_iters,
                                                        inverse=inverse)

        if inverse:
            return z
        else:
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)


class ConditionalLogisticMixtureAffineCouplingBijection(ConditionalCouplingBijection):

    def __init__(self, coupling_net, num_mixtures, context_net=None, split_dim=1, num_condition=None, scale_fn=lambda s: torch.exp(s)):
        super(ConditionalLogisticMixtureAffineCouplingBijection, self).__init__(
            coupling_net=coupling_net, context_net=context_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()
        assert callable(scale_fn)
        self.scale_fn = scale_fn

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures + 2
    
    def forward(self, x, context):
        if self.context_net: context = self.context_net(context)
        id, x2 = self.split_input(x)

        elementwise_params = self.coupling_net(id, context)

        z2, ldj = self._elementwise_forward(x2, elementwise_params)
        z = torch.cat([id, z2], dim=self.split_dim)
        return z, ldj

    def inverse(self, z, context):
        if self.context_net: context = self.context_net(context)
        id, z2 = self.split_input(z)
        #context = torch.cat([id, context], dim=self.split_dim)
        #elementwise_params = self.coupling_net(context)
        elementwise_params = self.coupling_net(id, context)
        x2 = self._elementwise_inverse(z2, elementwise_params)
        x = torch.cat([id, x2], dim=self.split_dim)
        return x


    def _elementwise(self, x, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        unconstrained_scale, shift, logit_weights, means, log_scales = get_flowpp_params(elementwise_params, num_mixtures=self.num_mixtures)
        scale = self.scale_fn(unconstrained_scale)
        log_scales = log_scales.clamp(min=-7)  # From the code in original Flow++ paper

        x, ldj_elementwise = logistic_mixture_transform(inputs=x,
                                                        logit_weights=logit_weights,
                                                        means=means,
                                                        log_scales=log_scales,
                                                        eps=self.eps,
                                                        max_iters=self.max_iters,
                                                        inverse=inverse)

        if inverse:
            z = (x - shift) / scale
            return z
        else:
            z = scale * x + shift
            logistic_ldj = sum_except_batch(ldj_elementwise)
            scale_ldj = sum_except_batch(torch.log(scale))
            ldj = logistic_ldj + scale_ldj
            
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)



class ConditionalCensoredLogisticMixtureCouplingBijection(ConditionalCouplingBijection):

    def __init__(self, coupling_net, num_mixtures, num_bins, context_net=None, split_dim=1, num_condition=None):
        super(ConditionalCensoredLogisticMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, context_net=context_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.num_bins = num_bins
        self.set_bisection_params()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = censored_logistic_mixture_transform(inputs=inputs,
                                                logit_weights=logit_weights,
                                                means=means,
                                                log_scales=log_scales,
                                                num_bins=self.num_bins,
                                                eps=self.eps,
                                                max_iters=self.max_iters,
                                                inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)
