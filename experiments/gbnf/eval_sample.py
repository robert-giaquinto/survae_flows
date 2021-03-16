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

path_samples = '{}/samples/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'], eval_args.seed)
if not os.path.exists(os.path.dirname(path_samples)):
    os.mkdir(os.path.dirname(path_samples))
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()

# save model samples
batch = next(iter(eval_loader))
if args.flow in ['sr', 'conditional']:
    imgs = batch[0][:eval_args.samples]
    context = batch[1][:eval_args.samples]
    samples = model.sample(context.to(device)).cpu().float() / (2**args.num_bits - 1)

    # save low-resolution samples too
    path_context_samples = '{}/samples/context_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'], eval_args.seed)
    context = context.cpu().float()
    if context.max().item() > 2:
        context /= (2**args.num_bits - 1)
    vutils.save_image(context, path_context_samples, nrow=eval_args.nrow)
else:
    samples = model.sample(eval_args.samples).cpu().float() / (2**args.num_bits - 1)
    imgs = batch[:eval_args.samples]
    
vutils.save_image(samples, path_samples, nrow=eval_args.nrow)
                
# save real samples too
path_true_samples = '{}/samples/true_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'], eval_args.seed)
imgs = imgs.cpu().float()
if imgs.max().item() > 2:
    imgs /= (2**args.num_bits - 1)
vutils.save_image(imgs, path_true_samples, nrow=eval_args.nrow)
