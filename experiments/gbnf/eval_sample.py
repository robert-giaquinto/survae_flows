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
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--nrow', type=int, default=8)
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

_, eval_loader, data_shape, cond_shape = get_data(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape, cond_shape=cond_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

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
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()

# save model samples
batch = next(iter(eval_loader))
if args.super_resolution:
    imgs = batch[0][:eval_args.samples]
    num_samples_or_context = batch[1][:eval_args.samples]
    path_context = f"{eval_args.model}/samples/context_e{checkpoint['current_epoch']}_s{eval_args.seed}.png"
    save_images(num_samples_or_context, path_context)
    num_samples_or_context = num_samples_or_context.to(device)
else:
    num_samples_or_context = eval_args.samples
    imgs = batch[:eval_args.samples]

temp_str = f"_t{int(100 * eval_args.temperature)}.png" if eval_args.temperature is not None else ".png"
if args.boosted_components > 1:
    for c in range(model.num_components):
        path_samples = f"{eval_args.model}/samples/sample_e{checkpoint['current_epoch']}_c{c}_s{eval_args.seed}" + temp_str
        samples = model.sample(num_samples_or_context, component=c, temperature=eval_args.temperature)
        save_images(samples, path_samples)
        
else:
    path_samples = f"{eval_args.model}/samples/sample_e{checkpoint['current_epoch']}_s{eval_args.seed}" + temp_str
    samples = model.sample(num_samples_or_context, temperature=eval_args.temperature)
    save_images(samples, path_samples)
                
# save real samples too
path_true_samples = f"{eval_args.model}/samples/true_e{checkpoint['current_epoch']}_s{eval_args.seed}.png"
save_images(imgs, path_true_samples)
