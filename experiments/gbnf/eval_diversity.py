import os
import math
import torch
import pickle
import argparse

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
parser.add_argument('--num_samples', type=int, default=128)
parser.add_argument('--num_repeats', type=int, default=1, help="Number of images to repeat the experiment over")
parser.add_argument('--temperature', nargs="*", type=float, default=[1.0])

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

args.batch_size = 1
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

assert len(eval_args.temperature) > 0
base_dir = os.path.join(f"{eval_args.model}", f"diversity/")
if not os.path.exists(base_dir): os.mkdir(base_dir)

for temperature in eval_args.temperature:
    out_dir = os.path.join(base_dir, f"{args.dataset}_temperature{ int(100 * temperature) }_seed{eval_args.seed}/")
    if not os.path.exists(os.path.dirname(out_dir)): os.mkdir(os.path.dirname(out_dir))
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    torch.manual_seed(eval_args.seed)

    avg_std = []
    for i, (_, x) in enumerate(eval_loader):
        if i == eval_args.num_repeats:
            break

        with torch.no_grad():
            sr_samples = model.sample(x.repeat(eval_args.num_samples, 1, 1, 1).to(device), temperature=temperature)
            std_per_pixel = torch.std(sr_samples.float().view(sr_samples.shape[0], -1), dim=0)
            avg_std_i = torch.mean(std_per_pixel).item()
            avg_std.append(avg_std_i)

    if eval_args.num_repeats > 1:
        sigma = torch.std(torch.tensor(avg_std)).item()
        mu = torch.tensor(avg_std).mean().item()
        diversity = f"Diversity: {mu:.2f} (+/- {sigma:.3f})"
    else:
        diversity = f"Diversity: {avg_std[0]:.2f}"

    print(f"Temperature {int(100 * temperature)}, " + diversity)
    out_fname = os.path.join(out_dir, f"diversity_repeats{eval_args.num_repeats}_samples{eval_args.num_samples}.txt")
    with open(out_fname, 'w') as f:
        f.write(diversity)


