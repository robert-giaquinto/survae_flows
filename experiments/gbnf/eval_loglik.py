import os
import math
import torch
import pickle
import argparse
import torchvision.utils as vutils
from survae.utils import dataset_elbo_bpd, dataset_iwbo_bpd, dataset_cond_elbo_bpd, dataset_cond_iwbo_bpd

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model import get_model, get_model_id, add_model_args
from survae.distributions import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--kbs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--seed', type=int, default=0)
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

##################
## Specify data ##
##################

eval_loader, data_shape, cond_shape = get_data(args, eval_only=True)

# Adjust args
args.batch_size = eval_args.batch_size

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape, cond_shape=cond_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

# Load checkpoint
exp.checkpoint_load('{}/check/'.format(more_args.model), device=more_args.new_device)

# modify model
if more_args.new_device is not None:
    exp.model.to(torch.device(more_args.new_device))

exp.eval_fn()
