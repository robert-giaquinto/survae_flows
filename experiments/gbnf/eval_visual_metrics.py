import os
import math
import torch
import pickle
import argparse
import torchvision.utils as vutils

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model import get_model, get_model_id, add_model_args
from survae.distributions import DataParallelDistribution

from survae.utils import PerceptualQuality, evaluate_perceptual_quality, format_metrics


###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--temperature', type=float, default=None)
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

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape, cond_shape=cond_shape)

if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

##############
## Evaluate ##
##############

metrics = evaluate_perceptual_quality(model, eval_loader, temperature=eval_args.temperature, device=device, sr_scale_factor=args.sr_scale_factor)

path_pc = f"{eval_args.model}/perceptual_quality/visual_metrics_ep{checkpoint['current_epoch']}_temp{int(100*eval_args.temperature)}_seed{eval_args.seed}.txt"
if not os.path.exists(os.path.dirname(path_pc)):
    os.mkdir(os.path.dirname(path_pc))

with open(path_pc, 'w') as f:
    f.write(format_metrics(metrics))
