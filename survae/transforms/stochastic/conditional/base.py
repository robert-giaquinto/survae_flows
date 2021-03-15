from survae.transforms import ConditionalTransform


class ConditionalStochasticTransform(ConditionalTransform):
    """Base class for ConditionalBijection"""

    has_inverse = True
    bijective = False
    stochastic_forward = True
    stochastic_inverse = True
