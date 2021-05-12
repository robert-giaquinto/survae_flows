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

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--prior_model', type=str, default=None)
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--nrow', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--temperature', nargs="*", type=float, default=[1.0])
parser.add_argument('--prior_temperature', nargs="*", type=float, default=[1.0])
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)
path_prior_args = '{}/args.pickle'.format(eval_args.prior_model)
path_prior_check = '{}/check/checkpoint.pt'.format(eval_args.prior_model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

with open(path_prior_args, 'rb') as f:
    prior_args = pickle.load(f)

args.batch_size = eval_args.samples
prior_args.batch_size = eval_args.samples

##################
## Specify data ##
##################

eval_loader, data_shape, cond_shape = get_data(args, eval_only=True)

####################
## Specify models ##
####################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# conditional model
model = get_model(args, data_shape=data_shape, cond_shape=cond_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model = model.eval()
print('Loaded weights for conditional model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

# prior model
prior_model = get_model(prior_args, data_shape=(data_shape[0], data_shape[1] // args.sr_scale_factor, data_shape[2] // args.sr_scale_factor))
if prior_args.parallel == 'dp':
    prior_model = DataParallelDistribution(prior_model)
prior_checkpoint = torch.load(path_prior_check, map_location=torch.device(device))
prior_model.load_state_dict(prior_checkpoint['model'])
prior_model = prior_model.to(device)
prior_model = prior_model.eval()
print('Loaded weights for prior model at {}/{} epochs'.format(prior_checkpoint['current_epoch'], prior_args.epochs))

############
## Sample ##
############

def save_images(imgs, file_path, num_bits=args.num_bits, nrow=eval_args.nrow):
    if not os.path.exists(os.path.dirname(file_path)):
        os.mkdir(os.path.dirname(file_path))
        
    out = imgs.cpu().float()
    if out.max().item() > 2:
        out /= (2**num_bits - 1)
            
    vutils.save_image(out, file_path, nrow=nrow)


assert len(eval_args.temperature) > 0

out_dir = os.path.join(f"{eval_args.model}", f"joint_samples/seed{eval_args.seed}/")
if not os.path.exists(os.path.dirname(os.path.dirname(out_dir))):
    os.mkdir(os.path.dirname(os.path.dirname(out_dir)))

if not os.path.exists(os.path.dirname(out_dir)):
    os.mkdir(os.path.dirname(out_dir))

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for prior_temperature in eval_args.prior_temperature:
    torch.manual_seed(eval_args.seed)
    
    # sample low-resolution from prior
    low_res = prior_model.sample(eval_args.samples, temperature=prior_temperature)
    path_lr = os.path.join(out_dir, f"big_prior_low_resolution_e{prior_checkpoint['current_epoch']}_temperature{int(100 * prior_temperature)}.png")
    save_images(low_res, path_lr)
    
    big_lr = torch.repeat_interleave(torch.repeat_interleave(low_res, args.sr_scale_factor, dim=2), args.sr_scale_factor, dim=3)
    path_big_lr = os.path.join(out_dir, f"prior_low_resolution_e{prior_checkpoint['current_epoch']}_temperature{int(100 * prior_temperature)}.png")
    save_images(big_lr, path_big_lr)

    for temperature in eval_args.temperature:
        
        # sample low-resolution from conditional
        high_res = model.sample(low_res, temperature=temperature)
        path_hr = os.path.join(out_dir, f"high_resolution_e{checkpoint['current_epoch']}_temperature{int(100 * temperature)}_prior{int(100 * prior_temperature)}.png")
        save_images(high_res, path_hr)

        
