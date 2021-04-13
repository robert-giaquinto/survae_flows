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
parser.add_argument('--num_batches', type=int, default=8)
parser.add_argument('--num_samples', type=int, default=11)
parser.add_argument('--image1_id', type=int, default=None)
parser.add_argument('--image2_id', type=int, default=None)
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

args.batch_size = 1
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

def save_images(imgs, file_path, num_bits=args.num_bits, nrow=1):
    if not os.path.exists(os.path.dirname(file_path)):
        os.mkdir(os.path.dirname(file_path))
        
    out = imgs.cpu().float()
    if out.max().item() > 2:
        out /= (2**num_bits - 1)

    nrow = imgs.size(0)
    vutils.save_image(out, file_path, nrow=nrow)
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()


if not os.path.exists(os.path.join(f"{eval_args.model}", "interpolation/")):
    os.mkdir(os.path.join(f"{eval_args.model}", "interpolation/"))

if eval_args.image1_id is not None and eval_args.image2_id is not None:
    # get images to interpolate between
    
    # map images to z's

    # interpolate samples between z's

    raise NotImplementedError
    
else:

    for batch_id in range(eval_args.num_batches):

        if args.super_resolution:
            batch = next(iter(eval_loader))            
            imgs = batch[0]
            context = batch[1]
            path_context = os.path.join(f"{eval_args.model}", f"interpolation/batch{batch_id}/context_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
            save_images(context, path_context)

            if args.boosted_components > 1:
                for c in range(model.num_components):
                    path_samples = os.path.join(f"{eval_args.model}", f"interpolation/batch{batch_id}/sample_e{checkpoint['current_epoch']}_c{c}_s{eval_args.seed}.png")
                    samples = model.interpolate(num_samples=eval_args.num_samples, context=context.to(device), component=c)
                    save_images(samples, path_samples)
        
            else:
                path_samples = os.path.join(f"{eval_args.model}", f"interpolation/batch{batch_id}/sample_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
                samples = model.interpolate(num_samples=eval_args.num_samples, context=context.to(device))
                save_images(samples, path_samples)

            # save real samples too
            path_true_samples = os.path.join(f"{eval_args.model}", f"interpolation/batch{batch_id}/true_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
            save_images(imgs, path_true_samples)
            
        else:
            if args.boosted_components > 1:
                for c in range(model.num_components):
                    path_samples = os.path.join(f"{eval_args.model}", f"interpolation/batch{batch_id}/sample_e{checkpoint['current_epoch']}_c{c}_s{eval_args.seed}.png")
                    samples = model.interpolate(num_samples=eval_args.num_samples, component=c)
                    save_images(samples, path_samples)
        
            else:
                path_samples = os.path.join(f"{eval_args.model}", f"interpolation/batch{batch_id}/sample_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
                samples = model.interpolate(num_samples=eval_args.num_samples)
                save_images(samples, path_samples)




                
