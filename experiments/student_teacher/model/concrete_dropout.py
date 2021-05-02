import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable

import numpy as np

class DropoutNet(nn.Module):
    """
    Concrete Dropout model: https://arxiv.org/abs/1705.07832
    
    Source code from:
    https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
    """
    def __init__(self, input_size, output_size, hidden_units, weight_regularizer, dropout_regularizer):
        super(DropoutNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, hidden_units)

        self.linear4_mu = nn.Linear(hidden_units, output_size)
        self.linear4_logvar = nn.Linear(hidden_units, output_size)

        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = ConcreteDropout(weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=weight_regularizer,
                                                 dropout_regularizer=dropout_regularizer)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        regularization = torch.empty(5, device=x.device)
        
        x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.linear1, self.relu))
        x2, regularization[1] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.relu))
        x3, regularization[2] = self.conc_drop3(x2, nn.Sequential(self.linear3, self.relu))

        mean, regularization[3] = self.conc_drop_mu(x3, self.linear4_mu)
        log_var, regularization[4] = self.conc_drop_logvar(x3, self.linear4_logvar)

        return mean, log_var, regularization.sum()

    def loss(self, x, y):
        mean, log_var, regularization = self.forward(x)
        loss = self.heteroscedastic_loss(y, mean, log_var) + regularization
        return loss
    
    @staticmethod
    def heteroscedastic_loss(true, mean, log_var):
        precision = torch.exp(-log_var)
        return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)



class ConcreteDropout(nn.Module):
    """
    Concrete Dropout model: https://arxiv.org/abs/1705.07832
    
    Source code from:
    https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
    """
    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)
        
        out = layer(self._concrete_dropout(x, p))
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x

    
