import torch
from survae.utils import sum_except_batch
from survae.transforms.bijections.functional.mixtures import gaussian_mixture_transform, logistic_mixture_transform, censored_logistic_mixture_transform
from survae.transforms.bijections.functional.mixtures import get_mixture_params, get_flowpp_params
from survae.transforms.bijections.coupling import CouplingBijection
from survae.transforms.bijections.elementwise_nonlinear import SigmoidInverse


class GaussianMixtureCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_mixtures, split_dim=1, num_condition=None):
        super(GaussianMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
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


class LogisticMixtureCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_mixtures, split_dim=1, num_condition=None):
        super(LogisticMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
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

        x = logistic_mixture_transform(inputs=inputs,
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


class LogisticMixtureAffineCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_mixtures, split_dim=1, num_condition=None, scale_fn=lambda s: torch.exp(s)):
        super(LogisticMixtureAffineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_mixtures = num_mixtures
        self.set_bisection_params()
        assert callable(scale_fn)
        self.scale_fn = scale_fn
        #self.sigmoid_inv = SigmoidInverse()

    def set_bisection_params(self, eps=1e-10, max_iters=100):
        self.max_iters = max_iters
        self.eps = eps

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures + 2

    def _elementwise_forward(self, x, elementwise_params):
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
                                                        inverse=False)

        # logistic inverse transform
        #x, sigmoid_ldj = self.sigmoid_inv(x)

        # affine transformation
        z = scale * x + shift
        logistic_ldj = sum_except_batch(ldj_elementwise)
        scale_ldj = sum_except_batch(torch.log(scale))
        ldj = logistic_ldj + scale_ldj #+ sigmoid_ldj
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift, logit_weights, means, log_scales = get_flowpp_params(elementwise_params, num_mixtures=self.num_mixtures)
        scale = self.scale_fn(unconstrained_scale)
        log_scales = log_scales.clamp(min=-7)  # From the code in original Flow++ paper
        x = (z - shift) / scale
        #x = torch.sigmoid(x)
        x = x.clamp(1e-5, 1.0 - 1e-5)
        x = logistic_mixture_transform(inputs=x,
                                       logit_weights=logit_weights,
                                       means=means,
                                       log_scales=log_scales,
                                       eps=self.eps,
                                       max_iters=self.max_iters,
                                       inverse=True)
        return x




class CensoredLogisticMixtureCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_mixtures, num_bins, split_dim=1, num_condition=None):
        super(CensoredLogisticMixtureCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
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
