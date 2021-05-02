import torch
import torchvision.utils as vutils
import numpy as np

from survae.distributions import DataParallelDistribution
from .utils import get_args_table, clean_dict

# Path
import os
import time
from survae.data.path import get_survae_path

# Experiment
from .base import BaseExperiment

# Logging frameworks
from torch.utils.tensorboard import SummaryWriter
import wandb

# Plot
import matplotlib.pyplot as plt

# SKLearn



class GaussianProcessExperiment(BaseExperiment):

    log_base = os.path.join(get_survae_path(), 'experiments/student_teacher/log')
    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'device',
                   'parallel',
                   'eval_every',
                   'num_flows', 'actnorm', 'scale_fn', 'hidden_units', 'range_flow', 'base_dist', 'affine', 'augment_size',
                   'pin_memory', 'num_workers']

    def __init__(self, args, data_id, model_id, model, teacher):

        # Edit args
        args.epoch = 1
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        # Move models
        self.teacher = teacher.to(args.device)
        self.teacher.eval()
        cond_id = args.cond_trans.lower()
        self.cond_size = 1 if cond_id.startswith('split') or cond_id.startswith('multiply') else 2

        seed_id = f"seed{args.seed}"
        arch_id = f"Gaussian_Process_{args.kernel}_kernel_a{int(args.gp_alpha)}_s{int(100*args.gp_length_scale)}"
        if args.name == "debug":
            log_path = os.path.join(
                self.log_base, "debug", model_id, data_id, cond_id, arch_id, seed_id, time.strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            log_path = os.path.join(
                self.log_base, model_id, data_id, cond_id, arch_id, seed_id, args.name)
        
        # Init parent
        super(GaussianProcessExperiment, self).__init__(model=model,
                                                        optimizer=None,
                                                        scheduler_iter=None,
                                                        scheduler_epoch=None,
                                                        log_path=log_path,
                                                        eval_every=args.eval_every)
        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.model_id = model_id
        self.data_id = data_id
        self.cond_id = cond_id
        self.arch_id = arch_id
        self.seed_id = seed_id

        # Init logging
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if args.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(
                args_dict).get_html_string(), global_step=0)

    def run(self):
        # Train
        train_dict = self.train_fn()
        self.log_train_metrics(train_dict)

        # Eval
        eval_dict = self.eval_fn()
        self.log_eval_metrics(eval_dict)

        # Log
        self.save_metrics()

        # Plotting
        self.plot_fn()

    def train_fn(self):
        self.teacher.eval()
        with torch.no_grad():
            y = self.teacher.sample(num_samples=self.args.train_samples)
            x = self.cond_fn(y)
            x = x.cpu().numpy()
            y = y.cpu().numpy()

        self.model.fit(x, y)
        nll = self.model.log_marginal_likelihood()
        r2 = self.model.score(x, y)
        print(f"Baseline: Log-marginal Likelihood={nll}, R-squared={r2}")
        return {'nll': nll, 'rsquared': r2}

    def eval_fn(self):
        K_test = 3 # number of MC samples
        
        self.teacher.eval()
        with torch.no_grad():
            y = self.teacher.sample(num_samples=self.args.test_samples)
            x = self.cond_fn(y)
            x = x.cpu().numpy()
            y = y.cpu().numpy()
            MC_samples = [self.model.predict(x, return_std=True) for _ in range(K_test)]

        means = np.stack([tup[0] for tup in MC_samples])
        logvar = np.stack([np.tile(tup[1], (2,1)).T for tup in MC_samples])

        test_ll = -0.5 * np.exp(-logvar) * (means - y.squeeze())**2.0 - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
        test_ll = np.sum(np.sum(test_ll, -1), -1)
        test_ll = logsumexp(test_ll) - np.log(K_test)
        #pppp = test_ll / self.args.test_samples  # per point predictive probability
        rmse = np.mean((np.mean(means, 0) - y.squeeze())**2.0)**0.5

        r2 = self.model.score(x, y)
        print(f"Baseline: R-squared={r2}, rmse={rmse}, likelihood={test_ll}")
        return {'rsquared': r2, 'rmse': rmse, 'lhood': test_ll}

    def plot_fn(self):
        plot_path = os.path.join(self.log_path, "samples/")
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        if self.args.dataset == 'face_einstein':
            bounds = [[0, 1], [0, 1]]
        else:
            bounds = [[-4, 4], [-4, 4]]

        # plot true data
        test_data = self.teacher.sample(num_samples=self.args.test_samples).data.numpy()
        plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
        plt.hist2d(test_data[...,0], test_data[...,1], bins=256, range=bounds)
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plt.axis('off')
        plt.savefig(os.path.join(plot_path, f'teacher.png'), bbox_inches='tight', pad_inches=0)

        # Plot samples while varying the context
        with torch.no_grad():
            y = self.teacher.sample(num_samples=self.args.num_samples)
            x = self.cond_fn(y).cpu().numpy()
        y_mean, y_std = self.model.predict(x, return_std=True)
        temperature = 0.4
        samples = [np.random.normal(y_mean[:, i], y_std * temperature, self.args.num_samples).T[:, np.newaxis] \
                   for i in range(y_mean.shape[1])]
        samples = np.hstack(samples)
        plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
        plt.hist2d(samples[...,0], samples[...,1], bins=256, range=bounds)
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plt.axis('off')
        plt.savefig(os.path.join(plot_path, f'varying_context_flow_samples.png'), bbox_inches='tight', pad_inches=0)
        
        # Plot density
        xv, yv = torch.meshgrid([torch.linspace(bounds[0][0], bounds[0][1], self.args.grid_size), torch.linspace(bounds[1][0], bounds[1][1], self.args.grid_size)])
        y = torch.cat([xv.reshape(-1,1), yv.reshape(-1,1)], dim=-1)
        x = self.cond_fn(y).numpy()
        means, logvar = self.model.predict(x, return_std=True)
        logvar = np.tile(logvar, (2,1)).T
        logprobs = -0.5 * np.exp(-logvar) * (means - y.numpy())**2.0 - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
        logprobs = np.sum(logprobs, -1)
        logprobs = logprobs - logprobs.max()
        probs = np.exp(logprobs)
        plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
        plt.pcolormesh(xv, yv, probs.reshape(xv.shape), shading='auto')
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plt.axis('off')
        plt.clim(0,self.args.clim)
        plt.savefig(os.path.join(plot_path, f'varying_context_flow_density.png'), bbox_inches='tight', pad_inches=0)

        # plot samples with fixed context
        for i, x in enumerate(self.context_permutations()):
            x = x.view((1, self.cond_size)).cpu().numpy()
            y_mean, y_std = self.model.predict(x, return_std=True)
            temperature = 1.5
            samples = np.hstack([np.random.normal(y_mean[:, i], y_std * temperature, self.args.num_samples).T[:, np.newaxis] \
                                 for i in range(y_mean.shape[1])])
            #samples = self.model.sample_y(x, n_samples=self.args.num_samples).T[..., 0]
            plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
            plt.hist2d(samples[...,0], samples[...,1], bins=256, range=bounds)
            plt.xlim(bounds[0])
            plt.ylim(bounds[1])
            plt.axis('off')
            plt.savefig(os.path.join(plot_path, f'fixed_context_flow_samples_{i}.png'), bbox_inches='tight', pad_inches=0)

            # Plot density
            xv, yv = torch.meshgrid([torch.linspace(bounds[0][0], bounds[0][1], self.args.grid_size), torch.linspace(bounds[1][0], bounds[1][1], self.args.grid_size)])
            y = torch.cat([xv.reshape(-1,1), yv.reshape(-1,1)], dim=-1).cpu().numpy()
            means, logvar = self.model.predict(x, return_std=True)
            logvar = np.tile(logvar, (2,1)).T
            logprobs = -0.5 * np.exp(-logvar) * (means - y)**2.0 - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
            logprobs = np.sum(logprobs, -1)
            logprobs = logprobs - logprobs.max()
            probs = np.exp(logprobs)
            plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
            plt.pcolormesh(xv, yv, probs.reshape(xv.shape), shading='auto')
            plt.xlim(bounds[0])
            plt.ylim(bounds[1])
            plt.axis('off')
            plt.clim(0,self.args.clim)
            plt.savefig(os.path.join(plot_path, f'fixed_context_flow_density_{i}.png'), bbox_inches='tight', pad_inches=0)


def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max



