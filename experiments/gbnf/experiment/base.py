import os
import pickle
import torch
import numpy as np

from survae.utils import count_parameters

from .utils import get_args_table, get_metric_table


class BaseExperiment(object):

    def __init__(self, model, optimizer, scheduler_iter, scheduler_epoch, log_path, eval_every, check_every):

        # Objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = os.path.join(log_path, 'check')

        # Intervals
        self.eval_every = eval_every
        self.check_every = check_every

        # Initialize
        self.current_epoch = 0
        self.component_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []

        # early stopping counters
        self.best_loss = np.inf
        self.best_loss_epoch = 0

        # automatic mixed precision
        self.amp = False
        self.scaler = None

        self.sample_batch = None

    def train_fn(self, epoch):
        raise NotImplementedError()

    def eval_fn(self, epoch):
        raise NotImplementedError()

    def log_fn(self, epoch, train_dict, eval_dict):
        raise NotImplementedError()

    def sample_fn(self):
        raise NotImplementedError()

    def stop_early(self, loss_dict, epoch):
        """
        Check if model meets early stopping criteria
        Returns: a tuple of two flags
            [T/F the model stopped improving, T/F the model improved this iteration]
        """
        return False, True
        
    def log_train_metrics(self, train_dict):
        if len(self.train_metrics)==0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics)==0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)

    def log_epoch(self, title, loss_count, data_count, loss_sum):
        print('{}. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
            title, self.current_epoch+1, self.args.epochs, loss_count, data_count, loss_sum/loss_count), end='\r')

    def create_folders(self):
        # Create log folder
        os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
        if self.check_every is not None:
            os.makedirs(self.check_path)
            print("Storing checkpoints in:", self.check_path)

    def save_args(self, args):
        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path, 'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def save_architecture(self):
        if self.args.name == "debug":
            print(self.model)
            print(f"\nNumber of trainable model parameters: {count_parameters(self.model)}\n")
            
        with open(os.path.join(self.log_path, 'architecture.txt'), "w") as f:
            f.write(str(self.model))
            f.write(f"\nNumber of trainable model parameters: {count_parameters(self.model)}\n")
            f.write(f"\nTotal number of model parameters: {sum(p.numel() for p in self.model.parameters())}\n")
            if hasattr(self.model, 'flow_shape'):
                f.write(f"\nOutput latent space shape: {self.model.flow_shape}\n")

    def save_metrics(self):
        # Save metrics
        with open(os.path.join(self.log_path,'metrics_train.pickle'), 'wb') as f:
            pickle.dump(self.train_metrics, f)
        with open(os.path.join(self.log_path,'metrics_eval.pickle'), 'wb') as f:
            pickle.dump(self.eval_metrics, f)

        # Save metrics table
        metric_table = get_metric_table(self.train_metrics, epochs=list(range(1, self.current_epoch+2)))
        with open(os.path.join(self.log_path,'metrics_train.txt'), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(self.eval_metrics, epochs=[e+1 for e in self.eval_epochs])
        with open(os.path.join(self.log_path,'metrics_eval.txt'), "w") as f:
            f.write(str(metric_table))

    def checkpoint_save(self):
        checkpoint = {'current_epoch': self.current_epoch,
                      'component_epoch': self.component_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'eval_epochs': self.eval_epochs,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      "scaler": self.scaler.state_dict() if self.amp else None,
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None}
        torch.save(checkpoint, os.path.join(self.check_path, 'checkpoint.pt'))

    def checkpoint_load(self, check_path, device=None):
        if device is not None:
            device = torch.device(device)
            checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pt'), map_location=device)
        else:
            checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pt'))

        self.current_epoch = checkpoint['current_epoch']
        self.component_epoch = checkpoint['component_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
        self.best_loss = np.inf
        self.best_loss_epoch = checkpoint['current_epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.amp: self.scaler.load_state_dict(checkpoint['scaler'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

    def run(self, epochs):
        for epoch in range(self.current_epoch, epochs):

            # Train
            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            # Eval
            if (epoch+1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch)
                self.log_eval_metrics(eval_dict)
                self.eval_epochs.append(epoch)
                converged, improved = self.stop_early(eval_dict, epoch)
                self.sample_fn()
            else:
                eval_dict = None
                converged = False
                improved = False

            # Log
            self.save_metrics()
            self.log_fn(epoch, train_dict, eval_dict)

            # Checkpoint
            self.current_epoch += 1
            if (self.check_every > 0 and (epoch+1) % self.check_every == 0) or improved:
                self.checkpoint_save()

            # Early stopping
            if converged:
                break

        # Sampling
        self.sample_fn()
