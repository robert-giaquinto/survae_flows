import torch


def get_mixture_params(params, num_mixtures):
    '''Get parameters for mixture transforms.'''
    assert params.shape[-1] == 3 * num_mixtures

    unnormalized_weights = params[..., :num_mixtures]
    means = params[..., num_mixtures:2*num_mixtures]
    log_scales = params[..., 2*num_mixtures:3*num_mixtures]

    return unnormalized_weights, means, log_scales


def get_flowpp_params(params, num_mixtures):
    '''
    Get parameters for Flow++'s logistic mixture transforms along with scale and shift.
    '''
    assert params.shape[-1] == 2 + 3 * num_mixtures

    s = params[..., 0]
    t = params[..., 1]
    unnormalized_weights = params[..., 2:(num_mixtures+2)]
    means = params[..., (num_mixtures+2):(2*num_mixtures+2)]
    log_scales = params[..., (2*num_mixtures+2):(3*num_mixtures+2)]

    return s, t, unnormalized_weights, means, log_scales
