import pickle
import torch
import torch.nn as nn
import copy

from survae.flows import ConditionalBoostedFlow

from model.init_model import init_model
from model.dequantization_flow import DequantizationFlow
from model.coupling import Coupling, MixtureCoupling


class ConditionalGBNF(ConditionalBoostedFlow):
    """
    Basic initializer for a gradient boosted normalizing flow

    TODO allow for args to be lists, so that each flow in the GBNF can be non-homogenous
    """

    def __init__(self, data_shape, cond_shape, num_bits, args):

        flows = []
        for c in range(args.boosted_components):
            if args.pretrained_model is not None and c == 0:
                path_args = '{}/args.pickle'.format(args.pretrained_model)
                path_check = '{}/check/checkpoint.pt'.format(args.pretrained_model)
                with open(path_args, 'rb') as f:
                    pretrained_args = pickle.load(f)

                flow = init_model(args=pretrained_args, data_shape=data_shape, cond_shape=cond_shape)
            else:
                flow_args = copy.deepcopy(args)
                flow_args.boosted_components = 1
                flow = init_model(args=flow_args, data_shape=data_shape, cond_shape=cond_shape)
                
            flows.append(flow)
        

        super(ConditionalGBNF, self).__init__(flows=flows, args=args)
