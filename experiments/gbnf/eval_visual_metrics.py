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
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--temperature', nargs="*", type=float, default=[1.0])
parser.add_argument('--resize_hw', type=int, default=None)

# set5 and set14 params
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--bicubic_lr', type=eval, default=False)
parser.add_argument('--crop', type=str, default=None, choices=[None, 'random', 'center'])

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

##############
## Evaluate ##
##############

pq = PerceptualQuality(device=device, num_bits=args.num_bits, sr_scale_factor=args.sr_scale_factor)

dataset_label = "" if eval_args.dataset is None else f"_{eval_args.dataset}"
for interpolation_method in ['nearest', 'bicubic']:
    metrics = pq.evaluate(model=interpolation_method, data_loader=eval_loader, temperature=None)
    print(pq.format_metrics(metrics))
    path_pc = f"{eval_args.model}/perceptual_quality/visual_metrics_{interpolation_method}_seed{eval_args.seed}{dataset_label}.txt"
    if not os.path.exists(os.path.dirname(path_pc)):
        os.mkdir(os.path.dirname(path_pc))

    with open(path_pc, 'w') as f:
        f.write(pq.format_metrics(metrics))


for temperature in eval_args.temperature:
    torch.manual_seed(eval_args.seed)
    metrics = pq.evaluate(model=model, data_loader=eval_loader, temperature=temperature)
    #print(f"Temperature: {temperature}")
    print(pq.format_metrics(metrics))
    
    path_pc = f"{eval_args.model}/perceptual_quality/visual_metrics_ep{checkpoint['current_epoch']}_temp{int(100*temperature)}_seed{eval_args.seed}{dataset_label}.txt"
    if not os.path.exists(os.path.dirname(path_pc)):
        os.mkdir(os.path.dirname(path_pc))

    with open(path_pc, 'w') as f:
        f.write(pq.format_metrics(metrics))
