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
parser.add_argument('--temperature', nargs="*", type=float, default=[1.0])
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

args.batch_size = eval_args.samples

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
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()

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


batch = next(iter(eval_loader))
assert len(eval_args.temperature) > 0
for temperature in eval_args.temperature:
    torch.manual_seed(eval_args.seed)

    out_dir = os.path.join(f"{eval_args.model}", f"samples/temperature{int(100 * temperature)}/")
    if not os.path.exists(os.path.dirname(out_dir)): os.mkdir(os.path.dirname(out_dir))
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    # save model samples
    if args.super_resolution:
        imgs = batch[0]  #[:eval_args.samples]
        num_samples_or_context = batch[1]  #[:eval_args.samples]
        path_context = os.path.join(out_dir, f"context_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
        save_images(num_samples_or_context, path_context)
        num_samples_or_context = num_samples_or_context.to(device)
        
        scale = args.sr_scale_factor
        big_lr = torch.repeat_interleave(torch.repeat_interleave(batch[1], scale, dim=2), scale, dim=3)
        path_big_lr = os.path.join(out_dir, f"big_lowres_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
        save_images(big_lr, path_big_lr)
    
    else:
        num_samples_or_context = eval_args.samples
        imgs = batch  #[:eval_args.samples]

    if args.boosted_components > 1:
        for c in range(model.num_components):
            path_samples = os.path.join(out_dir, f"sample_e{checkpoint['current_epoch']}_c{c}_s{eval_args.seed}.png")
            samples = model.sample(num_samples_or_context, component=c, temperature=temperature)
            save_images(samples, path_samples)
        
    else:
        path_samples = os.path.join(out_dir, f"sample_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
        samples = model.sample(num_samples_or_context, temperature=temperature)
        save_images(samples, path_samples)
                
    # save real samples too
    path_true_samples = os.path.join(out_dir, f"true_e{checkpoint['current_epoch']}_s{eval_args.seed}.png")
    save_images(imgs, path_true_samples)
