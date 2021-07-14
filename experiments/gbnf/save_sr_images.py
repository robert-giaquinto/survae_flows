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

def save_images(imgs, file_path, num_bits=args.num_bits):
    if not os.path.exists(os.path.dirname(file_path)):
        os.mkdir(os.path.dirname(file_path))

    max_color = 2.0**num_bits - 1.0
    out = torch.clamp(imgs.cpu().float(), min=0, max=max_color) / max_color
    #if out.max().item() > 2:
    #out /= max_color
    vutils.save_image(out, file_path)


i = 0
assert len(eval_args.temperature) > 0
for t, temperature in enumerate(eval_args.temperature):
    torch.manual_seed(eval_args.seed)

    yhat_dir = os.path.join(f"{eval_args.model}", f"yhat_test/temperature{int(100 * temperature)}/")
    if not os.path.exists(os.path.dirname(yhat_dir)): os.mkdir(os.path.dirname(yhat_dir))
    if not os.path.exists(yhat_dir): os.mkdir(yhat_dir)

    if t == 0:
        y_dir = os.path.join(f"{eval_args.model}", f"y_test/")
        if not os.path.exists(os.path.dirname(y_dir)): os.mkdir(os.path.dirname(y_dir))
        if not os.path.exists(y_dir): os.mkdir(y_dir)

    for x in eval_loader:
        context = x[1].to(device)
        samples = model.sample(context, temperature=temperature)
        for y, yhat in zip(x[0], samples):
            save_images(yhat, os.path.join(yhat_dir, f"yhat_{i}.png"))
            if t == 0:
                save_images(y, os.path.join(y_dir, f"y_{i}.png"))

            i += 1

