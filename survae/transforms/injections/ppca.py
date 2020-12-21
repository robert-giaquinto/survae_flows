import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.transforms.bijections import Injection


class PPCA(Injection):
    """
    PPCA transformation for density estimation
    Args:
        base_dim: int, Dimension size of the base distribution (expected to be smaller the data dimension)
        data_dim: int, Dimension size of the data inputs.
    """
    def __init__(self, base_dim, data_dim, bias=True, orthogonal_init=True):
        super(PPCA, self).__init__()
        self.base_dim = base_dim
        self.data_dim = data.dim

        self.A = nn.Parameter(torch.Tensor(data_dim, base_dim))
        if bias:
            self.mu = nn.Parameter(torch.Tensor(data_dim))
        else:
            self.register_parameter('mu', None)

        self.reset_parameters(orthogonal_init)

    def reset_parameters(self, orthogonal_init):
        self.orthogonal_init = orthogonal_init

        if self.orthogonal_init:
            nn.init.orthogonal_(self.A)
        else:
            bound = 1.0 / np.sqrt(self.data_dim)
            nn.init.uniform_(self.A, -bound, bound)

        if self.mu is not None:
            nn.init.zeros_(self.mu)

    def forward(self, x):
        diag_cov_inv = torch.eye(self.data_dim)  # I for now, otherwise estimate from data?

        ATSA = torch.eye(self.base_dim) + (self.A.t() @ diag_cov_inv) @ self.A
        ATSA_inv = torch.inverse(ATSA)

        if x.dim() > 2:
            # flatten image to a vector
            x = torch.flatten(x, start_dim=1)

        z = torch.matmul(ATSA_inv @ A.t() @ diag_cov_inv, x)
        x_proj = torch.matmul(self.A, z) @ diag_cov_inv

        # todo: move this to where the model calculates log_prob
        log_hx = -0.5 * torch.sum(x.t() @ (torch.bmm(x, diag_cov_inv) - x_proj), dim=-1)
        log_hx -= 0.5 * torch.slogdet(ATSA)[1]
        #log_hx -= 0.5 * log_diag_cov.sum()
        log_hx -= 0.5 * math.log(2 * math.pi) * self.data_dim
        return z, log_hx



        z = F.linear(x, self.weight, self.bias)
        _, ldj = torch.slogdet(self.weight)
        ldj = ldj.expand([x.shape[0]])
        return z, ldj

    def inverse(self, z):
        weight_inv = torch.inverse(self.weight)
        if self.bias is not None: z = z - self.bias
        x = F.linear(z, weight_inv)
        return x
