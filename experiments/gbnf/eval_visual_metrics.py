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

from survae.utils import PerceptualQuality


###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--temperature', nargs="*", type=float, default=[1.0])
eval_args = parser.parse_args()

assert len(eval_args.temperature) > 0

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
device = 'cpu'
model = model.to(device)
model = model.eval()
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

##############
## Evaluate ##
##############


pq = PerceptualQuality(device=device)
for temperature in eval_args.temperature:
    torch.manual_seed(eval_args.seed)
    metrics = pq.evaluate(model, eval_loader, temperature=temperature, sr_scale_factor=args.sr_scale_factor)
    print(pq.format_metrics(metrics))
    
    path_pc = f"{eval_args.model}/perceptual_quality/visual_metrics_ep{checkpoint['current_epoch']}_temp{int(100*temperature)}_seed{eval_args.seed}.txt"
    if not os.path.exists(os.path.dirname(path_pc)):
        os.mkdir(os.path.dirname(path_pc))

    with open(path_pc, 'w') as f:
        f.write(pq.format_metrics(metrics))
