import torch
import torchvision.utils as vutils

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


class StudentExperiment(BaseExperiment):

    log_base = os.path.join(get_survae_path(), 'experiments/student_teacher/log')
    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'eval_every',
                   'device', 'parallel',
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 model, teacher,
                 optimizer, scheduler_iter, scheduler_epoch):

        # Edit args
        if args.eval_every is None or args.eval_every == 0:
            args.eval_every = args.epochs
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        aug_or_abs = 'abs' if args.augment_size == 0 else f"aug{args.augment_size}"
        hidden = '_'.join([str(u) for u in args.hidden_units])
        cond_id = args.cond_trans.lower()
        arch_id = f"flow_{aug_or_abs}_k{args.num_flows}_h{hidden}_{'affine' if args.affine else 'additive'}{'_actnorm' if args.actnorm else ''}"
        seed_id = f"seed{args.seed}"
        if args.name == "debug":
            log_path = os.path.join(
                self.log_base, "debug", model_id, data_id, cond_id, arch_id, seed_id, time.strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            log_path = os.path.join(
                self.log_base, model_id, data_id, cond_id, arch_id, seed_id, args.name)

        # Move models
        model = model.to(args.device)
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)
        
        # Init parent
        super(StudentExperiment, self).__init__(model=model,
                                                optimizer=optimizer,
                                                scheduler_iter=scheduler_iter,
                                                scheduler_epoch=scheduler_epoch,
                                                log_path=log_path,
                                                eval_every=args.eval_every)
        # student teacher args
        teacher = teacher.to(args.device)
        self.teacher = teacher
        self.teacher.eval()
        self.cond_size = 1 if cond_id.startswith('split') or cond_id.startswith('multiply') else 2
        
        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.model_id = model_id
        self.data_id = data_id
        self.cond_id = cond_id
        self.optim_id = optim_id
        self.arch_id = arch_id
        self.seed_id = seed_id

        # Init logging
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if args.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(
                args_dict).get_html_string(), global_step=0)

        if args.log_wandb:
            wandb.init(config=args_dict, project=args.project, id=args.name, dir=self.log_path)

        # training params
        self.max_grad_norm = args.max_grad_norm

        # automatic mixed precision
        # bigger changes need to make this work with dataparallel though (@autocast() decoration on each forward)
        pytorch_170 = int(str(torch.__version__)[2]) >= 7
        self.amp = args.amp and args.parallel != 'dp' and pytorch_170
        if self.amp:
            # only available in pytorch 1.7.0+
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # save model architecture for reference
        self.save_architecture()

    def log_fn(self, epoch, train_dict, eval_dict):
        # Tensorboard
        if self.args.log_tb:
            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)

        # Weights & Biases
        if self.args.log_wandb:
            for metric_name, metric_value in train_dict.items():
                wandb.log({'base/{}'.format(metric_name): metric_value}, step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    wandb.log({'eval/{}'.format(metric_name): metric_value}, step=epoch+1)

    def resume(self):
        resume_path = os.path.join(self.log_base, self.model_id, self.data_id, self.cond_id, self.arch_id, self.seed_id, self.args.resume, 'check')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]
                
            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(
                        epoch)]
            else:
                eval_dict = None
            
            self.log_fn(epoch, train_dict=train_dict, eval_dict=eval_dict)

    def run(self):
        if self.args.resume:
            self.resume()
            
        super(StudentExperiment, self).run(epochs=self.args.epochs)

    def train_fn(self, epoch):
        if self.amp:
           # use automatic mixed precision
           return self._train_amp(epoch)
        else:
            return self._train(epoch)

    def _train_amp(self, epoch):
        """
        Same training procedure, but uses half precision to speed up training on GPUs.
        Only works on SOME GPUs and the latest version of Pytorch.
        """
        self.model.train()
        self.teacher.eval()
        loss_sum = 0.0
        loss_count = 0
        iters_per_epoch = self.args.train_samples // self.args.batch_size
        for i in range(iters_per_epoch):

            batch_size = self.args.batch_size
            with torch.no_grad():
                y = self.teacher.sample(num_samples=self.args.batch_size)
                x = self.cond_fn(y)
                
            with torch.cuda.amp.autocast():
                loss = -self.model.log_prob(y, context=x).mean()

            # Scale loss and call backward() to create scaled gradients
            self.scaler.scale(loss).backward()

            if self.max_grad_norm > 0:
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Unscale gradients and call (or skip) optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_iter:
                self.scheduler_iter.step()

            self.optimizer.zero_grad(set_to_none=True)

            # accumulate loss and report
            loss_sum += loss.detach().cpu().item() * batch_size
            loss_count += batch_size
            self.log_epoch("Training", loss_count, self.args.train_samples, loss_sum)

        print('')
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        
        return {'nll': loss_sum/loss_count}

    def _train(self, epoch):
        self.model.train()
        self.teacher.eval()
        loss_sum = 0.0
        loss_count = 0
        iters_per_epoch = self.args.train_samples // self.args.batch_size
        for i in range(iters_per_epoch):

            with torch.no_grad():
                y = self.teacher.sample(num_samples=self.args.batch_size)
                x = self.cond_fn(y)

            self.optimizer.zero_grad()
            loss = -self.model.log_prob(y, context=x).mean()
            loss.backward()

            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()

            loss_sum += loss.detach().cpu().item() * self.args.batch_size
            loss_count += self.args.batch_size
            self.log_epoch("Training", loss_count, self.args.train_samples, loss_sum)

        print('')
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        
        return {'nll': loss_sum/loss_count}

    def eval_fn(self):
        self.model.eval()
        self.teacher.eval()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            iters_per_epoch = self.args.test_samples // self.args.batch_size
            for i in range(iters_per_epoch):
                y = self.teacher.sample(num_samples=self.args.batch_size)
                x = self.cond_fn(y)
                loss = -self.model.log_prob(y, context=x).mean()
                loss_sum += loss.detach().cpu().item() * self.args.batch_size
                loss_count += self.args.batch_size
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, NLL: {:.3f}'.format(
                    self.current_epoch+1, self.args.epochs, loss_count, self.args.test_samples, loss_sum/loss_count), end='\r')
            print('')
        return {'nll': loss_sum/loss_count}

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
            x = self.cond_fn(y)
            samples = self.model.sample(context=x, temperature=0.4)
            samples = samples.cpu().numpy()

        plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
        plt.hist2d(samples[...,0], samples[...,1], bins=256, range=bounds)
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plt.axis('off')
        plt.savefig(os.path.join(plot_path, f'varying_context_flow_samples.png'), bbox_inches='tight', pad_inches=0)

        # Plot density
        xv, yv = torch.meshgrid([torch.linspace(bounds[0][0], bounds[0][1], self.args.grid_size), torch.linspace(bounds[1][0], bounds[1][1], self.args.grid_size)])
        y = torch.cat([xv.reshape(-1,1), yv.reshape(-1,1)], dim=-1).to(self.args.device)
        x = self.cond_fn(y)
        with torch.no_grad():
            logprobs = self.model.log_prob(y, context=x)
            logprobs = logprobs - logprobs.max().item()
            
        plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
        plt.pcolormesh(xv, yv, logprobs.exp().reshape(xv.shape))
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plt.axis('off')
        print('Range:', logprobs.exp().min().item(), logprobs.exp().max().item())
        print('Limits:', 0.0, self.args.clim)
        plt.clim(0,self.args.clim)
        plt.savefig(os.path.join(plot_path, f'varying_context_flow_density.png'), bbox_inches='tight', pad_inches=0)

        # plot samples with fixed context
        for i, x in enumerate(self.context_permutations()):
            with torch.no_grad():
                #y = self.teacher.sample(num_samples=1).expand((self.args.num_samples, 2))
                samples = self.model.sample(context=x.expand((self.args.num_samples, self.cond_size)), temperature=1.5).cpu().numpy()

            plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
            plt.hist2d(samples[...,0], samples[...,1], bins=256, range=bounds)
            plt.xlim(bounds[0])
            plt.ylim(bounds[1])
            plt.axis('off')
            plt.savefig(os.path.join(plot_path, f'fixed_context_flow_samples_{i}.png'), bbox_inches='tight', pad_inches=0)

            # Plot density
            xv, yv = torch.meshgrid([torch.linspace(bounds[0][0], bounds[0][1], self.args.grid_size), torch.linspace(bounds[1][0], bounds[1][1], self.args.grid_size)])
            y = torch.cat([xv.reshape(-1,1), yv.reshape(-1,1)], dim=-1).to(self.args.device)
            with torch.no_grad():
                logprobs = self.model.log_prob(y, context=x.expand((y.size(0), self.cond_size)))
                logprobs = logprobs - logprobs.max().item()

            plt.figure(figsize=(self.args.pixels/self.args.dpi, self.args.pixels/self.args.dpi), dpi=self.args.dpi)
            plt.pcolormesh(xv, yv, logprobs.exp().reshape(xv.shape))
            plt.xlim(bounds[0])
            plt.ylim(bounds[1])
            plt.axis('off')
            print('Range:', logprobs.exp().min().item(), logprobs.exp().max().item())
            print('Limits:', 0.0, self.args.clim)
            plt.clim(0,self.args.clim)
            plt.savefig(os.path.join(plot_path, f'fixed_context_flow_density_{i}.png'), bbox_inches='tight', pad_inches=0)
    




