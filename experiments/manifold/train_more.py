import torch
import numpy as np
import time
import pickle
import argparse
from utils import set_seeds

from survae.distributions import ConvNormal2d, StandardNormal, StandardUniform

# Exp
from experiment.flow import FlowExperiment, add_exp_args

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model import get_model, get_model_id, add_model_args

# Optim
from optim import get_optim, get_optim_id, add_optim_args


###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--new_epochs', type=int)
parser.add_argument('--new_lr', type=float, default=None)
parser.add_argument('--new_device', type=str, default=None)
parser.add_argument('--new_batch_size', type=int, default=None)
parser.add_argument('--new_augmentation', type=str, default=None)
parser.add_argument('--base_distributions', type=str, default=None)
parser.add_argument('--freeze', type=eval default=None, help="True to keep layers of a pretrained model frozen, False to fine-tune")
add_data_args(parser)
more_args = parser.parse_args()

path_args = '{}/args.pickle'.format(more_args.model)
path_check = '{}/check/checkpoint.pt'.format(more_args.model)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

# Adjust args
args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
args.epochs = more_args.new_epochs
if more_args.new_augmentation is not None: args.augmentation = more_args.new_augmentation
if more_args.freeze is not None: args.freeze = more_args.freeze
args.resume = None
args.pretrained = False
if more_args.new_lr is not None: args.lr = more_args.new_lr
if more_args.new_batch_size is not None: args.batch_size = more_args.new_batch_size

# Store more_args
args.start_model = more_args.model
args.new_epochs = more_args.new_epochs
args.new_lr = more_args.new_lr if more_args.new_lr is not None else args.lr
args.new_augmentation = more_args.new_augmentation
if more_args.new_device is not None: args.device = more_args.new_device
if more_args.base_distributions is not None: args.base_distributions = more_args.base_distributions
if hasattr(args, 'amp') == False:
    args.amp = False
    args.scaler = None


##################
## Specify data ##
##################

train_loader, eval_loader, data_shape = get_data(args)
data_id = get_data_id(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
model_id = get_model_id(args)

#######################
## Specify optimizer ##
#######################

optimizer, _, _ = get_optim(args, model)
optim_id = 'more'

##############
## Training ##
##############

exp = FlowExperiment(args=args,
                     data_id=data_id,
                     model_id=model_id,
                     optim_id=optim_id,
                     train_loader=train_loader,
                     eval_loader=eval_loader,
                     model=model,
                     optimizer=optimizer,
                     scheduler_iter=None,
                     scheduler_epoch=None)


# Load checkpoint
exp.checkpoint_load('{}/check/'.format(more_args.model), device=more_args.new_device)

# modify model
if more_args.new_device is not None:
    exp.model.to(torch.device(more_args.new_device))
    
if more_args.base_distributions is not None:
    # for now just assume we're setting base to normal normal
    if args.compression != "vae":
        if more_args.base_distributions == "n":
            base_dist = StandardNormal(model.flow_shape)
        elif more_args.base_distributions == "c":
            base_dist = ConvNormal2d(model.flow_shape)
        elif more_args.base_distributions == "u":
            base_dist = StandardUniform(model.flow_shape)

        exp.model.base_dist = base_dist
    else:
        base_dist = []
        for i, d in enumerate(more_args.base_distributions):
            first_of_multiple = len(more_args.base_distributions) > 1 and i == 0
            s = model.flow_shape if first_of_multiple else (args.latent_size,)
                
            if d == "n":
                base_dist.append(StandardNormal(s))
            elif d == "c":
                base_dist.append(ConvNormal2d(s))
            elif d == "u":
                base_dist.append(StandardUniform(s))

            exp.model.base_dist = torch.nn.ModuleList(base_dist)

    print("Changed model's base distribution:\n", exp.model)

if more_args.new_lr is not None:
    # Adjust lr
    for param_group in exp.optimizer.param_groups:
        param_group['lr'] = more_args.new_lr

exp.run()
