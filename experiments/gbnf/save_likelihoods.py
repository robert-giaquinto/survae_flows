import os
import math
import torch
import pickle
import argparse
import numpy as np

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
parser.add_argument('--seed', type=int, default=0)

# set5 and set14 params
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--resize_hw', type=int, default=None)
parser.add_argument('--repeats', type=int, default=1, help="How many times the set5 or set14 dataset is repeated by the loader")
parser.add_argument('--bicubic_lr', type=eval, default=False)
parser.add_argument('--crop', type=str, default=None, choices=[None, 'random', 'center'])


eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

args.repeats = eval_args.repeats
args.bicubic_lr = eval_args.bicubic_lr
args.crop = eval_args.crop
if eval_args.dataset is not None:
    args.dataset = eval_args.dataset
    args.resize_hw = eval_args.resize_hw

##################
## Specify data ##
##################

eval_loader, data_shape, cond_shape = get_data(args, eval_only=True)

###################
## Specify model ##
###################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = get_model(args, data_shape=data_shape, cond_shape=cond_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)

checkpoint = torch.load(path_check, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model = model.eval()
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

############
## Sample ##
############

base_dir = os.path.join(f"{eval_args.model}", f"likelihoods/")
if not os.path.exists(base_dir): os.mkdir(base_dir)

lhoods = []
if args.super_resolution:
    for y, x in eval_loader:
        with torch.no_grad():
            lhood = model.log_prob(y.to(device), x.to(device))
            lhoods.append(lhood)
else:
    for x in eval_loader:
        with torch.no_grad():
            lhood = model.log_prob(x.to(device))
            lhoods.append(lhood)
    

lhoods = torch.cat(lhoods, dim=0).cpu().data.numpy()
print(f"Likelihoods shape: {lhoods.shape}")
np.savetxt(os.path.join(base_dir, f"likelihoods_{args.dataset}_seed{eval_args.seed}.txt"), lhoods, delimiter=',', fmt='%10.5f')
