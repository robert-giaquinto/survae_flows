import pickle
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

    TODO:
          1. Allow for args to be lists, so that each flow in the GBNF can be non-homogenous
          2. Allow a pretrained boosted model to be extended with more components

    """

    def __init__(self, data_shape, num_bits, args):

        flows = []
        for c in range(args.boosted_components):
            if args.pretrained_model is not None and c == 0:
                path_args = '{}/args.pickle'.format(args.pretrained_model)
                path_check = '{}/check/checkpoint.pt'.format(args.pretrained_model)
                with open(path_args, 'rb') as f:
                    pretrained_args = pickle.load(f)
                    pretrained_args.resume = None
                    if hasattr(pretrained_args, 'augment_size') == False:
                        pretrained_args.augment_size = 0

                flow = init_model(args=pretrained_args, data_shape=data_shape)
            else:
                flow_args = copy.deepcopy(args)
                flow_args.boosted_components = 1
                flow = init_model(args=flow_args, data_shape=data_shape)

            flows.append(flow)
        

        super(GBNF, self).__init__(flows=flows, args=args)

