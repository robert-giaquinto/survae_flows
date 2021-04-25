import torch
import argparse
from utils import set_seeds

# Exp
from experiment.base import add_exp_args
from experiment.student_experiment import StudentExperiment

# Data
from data.data import add_data_args

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

###################
## Specify model ##
###################

student, teacher, data_id = get_model(args)
model_id = get_model_id(args)
args.dataset = data_id

#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, student.parameters())
optim_id = get_optim_id(args)

##############
## Training ##
##############

exp = StudentExperiment(args=args,
                        data_id=data_id,
                        model_id=model_id,
                        optim_id=optim_id,
                        model=student,
                        teacher=teacher,
                        optimizer=optimizer,
                        scheduler_iter=scheduler_iter,
                        scheduler_epoch=scheduler_epoch)

exp.run()

