
import torch
import torch.nn as nn
import copy

from survae.flows import BoostedFlow

from model.init_model import init_model
from model.dequantization_flow import DequantizationFlow
from model.coupling import Coupling, MixtureCoupling


class GBNF(BoostedFlow):
    """
    Basic initializer for a gradient boosted normalizing flow

    TODO allow for args to be lists, so that each flow in the GBNF can be non-homogenous
    """

    def __init__(self, data_shape, num_bits, args):

        flows = []
        for c in range(args.boosted_components):
            flow_args = copy.deepcopy(args)
            flow_args.boosted_components = 1
            flow = init_model(args=flow_args, data_shape=data_shape)
            flows.append(flow)
        

        super(GBNF, self).__init__(flows=flows, args=args)
