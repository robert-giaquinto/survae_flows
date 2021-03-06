import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.nn.layers import LambdaLayer
from survae.nn.layers import act_module


class MLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_units, activation='relu', in_lambda=None, out_lambda=None):
        if isinstance(hidden_units, list) == False:
            hidden_units = [hidden_units]
        
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))

        if len(hidden_units) > 0:
            for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
                layers.append(nn.Linear(in_size, out_size))
                if activation is not None:
                    layers.append(act_module(activation))

            layers.append(nn.Linear(hidden_units[-1], output_size))
        else:
            layers.append(nn.Linear(input_size, output_size))

        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(MLP, self).__init__(*layers)

