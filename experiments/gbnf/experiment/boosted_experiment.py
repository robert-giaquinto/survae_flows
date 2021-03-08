import torch
import torchvision.utils as vutils
import math
import numpy as np

from survae.distributions import DataParallelDistribution
from survae.utils import elbo_bpd
from .utils import get_args_table, clean_dict

# Path
import os
import time
from survae.data.path import get_survae_path

# Experiment
from .base import BaseExperiment
from .flow_experiment import FlowExperiment

from experiments.gbnf.optim import get_optim

# Logging frameworks
from torch.utils.tensorboard import SummaryWriter
#import wandb

    
class BoostedFlowExperiment(FlowExperiment):

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):
            
        # Init parent
        super(BoostedFlowExperiment, self).__init__(args=args,
                                                    data_id=data_id, model_id=model_id, optim_id=optim_id,
                                                    train_loader=train_loader,
                                                    eval_loader=eval_loader,
                                                    model=model,
                                                    optimizer=optimizer,
                                                    scheduler_iter=scheduler_iter,
                                                    scheduler_epoch=scheduler_epoch)
        
        self.num_components = args.boosted_components
        self.epochs_per_component = self.args.epochs
        self.args.epochs = self.args.epochs * self.num_components
        self.component_epoch = 0

    def run(self):
        if self.args.resume:
            self.resume()

        while self.model.component < self.num_components:
            self.init_component()
            
            for epoch in range(self.component_epoch, self.epochs_per_component):

                # Train
                train_dict = self.train_fn(epoch)
                self.log_train_metrics(train_dict)

                # Eval
                if (epoch+1) % self.eval_every == 0:
                    eval_dict = self.eval_fn(epoch)
                    self.log_eval_metrics(eval_dict)
                    self.eval_epochs.append(epoch)
                    converged, improved = self.stop_early(eval_dict, epoch)
                else:
                    eval_dict = None
                    converged = False
                    improved = False

                # Log
                self.save_metrics()
                self.log_fn(epoch, train_dict, eval_dict)

                # Checkpoint
                self.current_epoch += 1
                self.component_epoch += 1
                if (self.check_every > 0 and (epoch+1) % self.check_every == 0) or improved:
                    self.checkpoint_save()

                # Early stopping
                if converged:
                    break

            # initialize training for next component
            if self.check_every == 0:
                self.resume()  # reload if using early stopping

            print(f"--- Boosting component {self.model.component + 1}/{self.num_components} complete ---")
            self.model.update_rho(self.train_loader)
            self.model.increment_component()
            self.component_epoch = 0
            self.optimizer, self.scheduler_iter, self.scheduler_epoch = get_optim(self.args, self.model)
            self.checkpoint_save()
            
        # Sampling
        self.sample_fn()

    def eval_fn(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            loss_sum2 = 0.0
            loss_sum3 = 0.0
            loss_count = 0
            for x in self.eval_loader:
                loss = -1.0 * self.model.mult_mixture_log_prob(x.to(self.args.device)).sum() / (math.log(2) * x.shape.numel())
                loss_sum += loss.detach().cpu().item() * len(x)

                loss2 = -1.0 * self.model.add_mixture_log_prob(x.to(self.args.device)).sum() / (math.log(2) * x.shape.numel())
                loss_sum2 += loss2.detach().cpu().item() * len(x)

                loss3 = -1.0 * self.model.approximate_mixture_log_prob(x.to(self.args.device)).sum() / (math.log(2) * x.shape.numel())
                loss_sum3 += loss3.detach().cpu().item() * len(x)

                loss_count += len(x)
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f} {:.3f} {:.3f}'.format(
                    self.current_epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count, loss_sum2/loss_count, loss_sum3/loss_count), end='\r')
            print('')
        return {'bpd': loss_sum/loss_count}

    # def log_epoch(self, title, loss_count, data_count, loss_sum):
    #     print('{}. Component: {}/{}, Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
    #         title, self.model.component + 1, self.num_components, self.current_epoch+1, self.args.epochs, loss_count, data_count, loss_sum/loss_count), end='\r')

    def sample_fn(self):
        if self.args.samples < 1:
            return
        
        self.model.eval()

        path_check = '{}/check/checkpoint.pt'.format(self.log_path)
        checkpoint = torch.load(path_check)

        for c in range(self.num_components):
            path_samples = '{}/samples/sample_ep{}_s{}_c{}.png'.format(self.log_path, checkpoint['current_epoch'], self.args.seed, c)
            if not os.path.exists(os.path.dirname(path_samples)):
                os.mkdir(os.path.dirname(path_samples))

            # save model samples
            samples = self.model.sample(self.args.samples, component=c).cpu().float() / (2**self.args.num_bits - 1)
            vutils.save_image(samples, path_samples, nrow=self.args.nrow)
                
        # save real samples too
        path_true_samples = '{}/samples/true_ep{}_s{}.png'.format(self.log_path, checkpoint['current_epoch'], self.args.seed)
        imgs = next(iter(self.eval_loader))[:self.args.samples]
        vutils.save_image(imgs.cpu().float(), path_true_samples, nrow=self.args.nrow)

    def init_component(self):
        self.best_loss = np.inf
        self.best_loss_epoch = 0

        for c in range(self.num_components):
            if c != self.model.component:
                self.optimizer.param_groups[c]['lr'] = 0.0

        for n, param in self.model.named_parameters():
            param.requires_grad = True if n.startswith(f"flows.{self.model.component}") else False

    def update_learning_rates(self):
        for c in range(self.num_components):
            self.optimizer.param_groups[c]['lr'] = self.args.lr if c == model.component else 0.0


