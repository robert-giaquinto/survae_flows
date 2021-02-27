import torch
import numpy as np
import time
import pickle
import argparse
from utils import set_seeds

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
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--freeze', type=eval, default=True)
parser.add_argument('--latent_size', type=int, default=196)

add_exp_args(parser)
more_args = parser.parse_args()
set_seeds(more_args.seed)

path_args = '{}/args.pickle'.format(more_args.model)
path_check = '{}/check/checkpoint.pt'.format(more_args.model)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

# Adjust args with new arguments and arguments missing in the pretrained model
args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
args.resume = None
args.compression = "pretrained"
args.exponential_lr = True
args.annealing_schedule = 0
args.load_pretrained weights = True

# Store changes to existing args from more_args
args.pretrained_model = more_args.model
args.new_epochs = more_args.epochs
args.new_lr = more_args.lr if more_args.lr is not None else args.lr

# update args with new inputs
args.epochs = more_args.epochs
if more_args.lr is not None: args.lr = more_args.lr
if more_args.batch_size is not None: args.batch_size = more_args.batch_size
if more_args.device is not None: args.device = more_args.device
args.freeze = more_args.freeze
args.latent_size = more_args.latent_size
args.vae_hidden_units = more_args.vae_hidden_units
args.vae_activation = more_args.vae_activation

args.max_grad_norm = more_args.max_grad_norm
args.early_stop = more_args.early_stop
args.save_samples = more_args.save_samples
args.log_tb = more_args.log_tb
args.log_wandb = more_args.log_wandb
args.amp = more_args.amp


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
optim_id = get_optim_id(args)

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

if more_args.new_lr is not None:
    # Adjust lr
    for param_group in exp.optimizer.param_groups:
        param_group['lr'] = more_args.new_lr

exp.run()
