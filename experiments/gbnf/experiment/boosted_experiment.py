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
        self.component_epoch = 0
        if args.pretrained_model is not None:
            self.args.epochs = self.args.epochs * (self.num_components - 1)
        else:
            self.args.epochs = self.args.epochs * self.num_components

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
                    self.sample_fn(components="c")
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
        self.sample_fn(components="1:c")

    def eval_fn(self, epoch):
        if self.args.super_resolution or self.args.conditional:
            return self._cond_eval_fn(epoch)
        else:
            return self._eval_fn(epoch)
    
    def _cond_eval_fn(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            approx_loss_sum = 0.0
            loss_count = 0
            for (x, context) in self.eval_loader:
                batch_size = len(x)
                context = context.to(self.args.device)
                x = x.to(self.args.device)
                
                loss = -1.0 * self.model.log_prob(x, context).sum() / (math.log(2) * x.shape.numel())
                loss_sum += loss.detach().cpu().item() * batch_size

                approx_loss = -1.0 * self.model.approximate_mixture_log_prob(x, context).sum() / (math.log(2) * x.shape.numel())
                approx_loss_sum += approx_loss.detach().cpu().item() * batch_size

                loss_count += batch_size
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}, aprx={:.3f}'.format(
                    self.current_epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count, approx_loss_sum/loss_count), end='\r')
            print('')
        return {'bpd': loss_sum/loss_count, 'bpd_aprx': approx_loss_sum/loss_count}

    def _eval_fn(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            approx_loss_sum = 0.0

            loss_count = 0
            for x in self.eval_loader:
                batch_size = len(x)
                x.to(self.args.device)

                loss = -1.0 * self.model.log_prob(x).sum() / (math.log(2) * x.shape.numel())
                loss_sum += loss.detach().cpu().item() * batch_size

                approx_loss = -1.0 * self.model.approximate_mixture_log_prob(x).sum() / (math.log(2) * x.shape.numel())
                approx_loss_sum += approx_loss.detach().cpu().item() * batch_size

                loss_count += batch_size
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}, aprx={:.3f}'.format(
                    self.current_epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count, approx_loss_sum/loss_count), end='\r')                
            print('')
        return {'bpd': loss_sum/loss_count, 'bpd_aprx': approx_loss_sum/loss_count}

    def sample_fn(self, components="1:c", sample_new_batch=False):
        if self.args.samples < 1:
            return
        
        self.model.eval()
        new_batch = self.sample_batch is None or sample_new_batch
        if new_batch:
            self.sample_batch = next(iter(self.eval_loader))

        if self.args.super_resolution or args.conditional:
            imgs = self.sample_batch[0][:self.args.samples]
            context = self.sample_batch[1][:self.args.samples]
            self._cond_sample_fn(context, components, save_context=new_batch)
        else:
            imgs = self.sample_batch[:self.args.samples]
            self._sample_fn(components)

        if new_batch:
            # save real samples
            path_true_samples = '{}/samples/true_te{}_s{}.png'.format(self.log_path, self.current_epoch, self.args.seed)
            self.save_images(imgs, path_true_samples)

    def _cond_sample_fn(self, context, components, save_context=True):
        if self.args.super_resolution and save_context:
            path_context = '{}/samples/context_te{}_s{}.png'.format(self.log_path, self.current_epoch, self.args.seed)
            self.save_images(context, path_context)

        if components == "1:c":
            # save samples from each component
            for c in range(self.num_components):
                path_samples = '{}/samples/sample_te{}_c{}_s{}.png'.format(self.log_path, self.current_epoch, c, self.args.seed)
                samples = self.model.sample(context.to(self.args.device), component=c)
                self.save_images(samples, path_samples)
        else:
            path_samples = '{}/samples/sample_c{}_ce{}_te{}_s{}.png'.format(
                self.log_path, self.model.component, self.component_epoch, self.current_epoch, self.args.seed)
            samples = self.model.sample(context.to(self.args.device), component=self.model.component)
            self.save_images(samples, path_samples)
            
    def _sample_fn(self, components):
        if components == "1:c":
            for c in range(self.num_components):
                path_samples = '{}/samples/sample_te{}_c{}_s{}.png'.format(self.log_path, self.current_epoch, c, self.args.seed)
                samples = self.model.sample(self.args.samples, component=c)
                self.save_images(samples, path_samples)
        else:
            path_samples = '{}/samples/sample_component{}_componentepoch{}_totalepochs{}_seed{}.png'.format(
                self.log_path, self.model.component, self.component_epoch, self.current_epoch, self.args.seed)
            samples = self.model.sample(self.args.samples, component=self.model.component)
            self.save_images(samples, path_samples)
                
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


