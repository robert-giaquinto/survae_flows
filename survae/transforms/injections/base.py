from survae.transforms import Transform


class Injection(Transform):
    """Base class for Injection"""

    bijective = False
    stochastic_forward = False
    stochastic_inverse = False
    lower_bound = False
