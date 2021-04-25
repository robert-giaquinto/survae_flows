import torch
import argparse
from utils import set_seeds

# Exp
from experiment.base import add_exp_args
from experiment.teacher_experiment import TeacherExperiment

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model import get_model, get_model_id, add_model_args

# Optim
from optim import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
set_seeds(args.seed)

##################
## Specify data ##
##################

train_loader, eval_loader = get_data(args)
data_id = get_data_id(args)

###################
## Specify model ##
###################

model = get_model(args)
model_id = get_model_id(args)

#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model.parameters())
optim_id = get_optim_id(args)

##############
## Training ##
##############

exp = TeacherExperiment(args=args,
                        data_id=data_id,
                        model_id=model_id,
                        optim_id=optim_id,
                        train_loader=train_loader,
                        eval_loader=eval_loader,
                        model=model,
                        optimizer=optimizer,
                        scheduler_iter=scheduler_iter,
                        scheduler_epoch=scheduler_epoch)

exp.run()

