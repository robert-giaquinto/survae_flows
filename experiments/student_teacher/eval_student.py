import os
import math
import torch
import pickle
import argparse

# Data
from data.data import add_data_args

# Model
from model.model import get_model, get_model_id
from model.baseline import get_baseline
from survae.distributions import DataParallelDistribution

# Optim
from optim import get_optim, get_optim_id, add_optim_args

# Experiment
from experiment.student_experiment import StudentExperiment
from experiment.dropout_experiment import DropoutExperiment
from experiment.gp_experiment import GaussianProcessExperiment


###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--model_type', type=str, default=None, choices=['flow', 'gp', 'dropout'])
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

################
## Experiment ##
################

if eval_args.model_type == "flow":
    student, teacher, data_id = get_model(args)
    model_id = get_model_id(args)
    args.dataset = data_id

    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, student.parameters())
    optim_id = get_optim_id(args)

    exp = StudentExperiment(args=args, data_id=data_id, model_id=model_id, optim_id=optim_id,
                            model=student,
                            teacher=teacher,
                            optimizer=optimizer,
                            scheduler_iter=scheduler_iter,
                            scheduler_epoch=scheduler_epoch)
else:
    student, teacher, data_id = get_baseline(args)
    model_id = get_model_id(args)
    args.dataset = data_id

    if args.baseline == "gp":
        exp = GaussianProcessExperiment(args=args, data_id=data_id,model_id=model_id,
                                        model=student,
                                        teacher=teacher)

    elif args.baseline == "dropout":
        optimizer, scheduler_iter, scheduler_epoch = get_optim(args, student.parameters())
        optim_id = get_optim_id(args)
        exp = DropoutExperiment(args=args, data_id=data_id, model_id=model_id, optim_id=optim_id,
                                model=student,
                                teacher=teacher,
                                optimizer=optimizer,
                                scheduler_iter=scheduler_iter,
                                scheduler_epoch=scheduler_epoch)

# Load checkpoint
exp.checkpoint_load('{}/check/'.format(more_args.model), device=more_args.new_device)


##############
## Evaluate ##
##############
exp.eval_fn()
