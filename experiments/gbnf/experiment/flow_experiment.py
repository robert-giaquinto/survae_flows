import torch
import torchvision.utils as vutils

from survae.distributions import DataParallelDistribution
from survae.utils import elbo_bpd, cond_elbo_bpd
from .utils import get_args_table, clean_dict

# Path
import os
import time
from survae.data.path import get_survae_path

# Experiment
from .base import BaseExperiment

# Logging frameworks
from torch.utils.tensorboard import SummaryWriter
#import wandb


def add_exp_args(parser):
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
    parser.add_argument('--amp', type=eval, default=False, help="Use automatic mixed precision")
    parser.add_argument('--resume', type=str, default=None)

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=0.0)
    parser.add_argument('--early_stop', type=int, default=0, help="Number of epochs with no improvement before stopping training")

    # sampling parameters
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--nrow', type=int, default=8)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=False)
    parser.add_argument('--log_wandb', type=eval, default=False)


class FlowExperiment(BaseExperiment):

    log_base = os.path.join(get_survae_path(), 'experiments/gbnf/log')
    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'check_every', 'eval_every',
                   'device', 'parallel',
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):

        # Edit args
        if args.eval_every is None or args.eval_every == 0:
            args.eval_every = args.epochs
        if args.check_every is None or args.check_every == 0:
            args.check_every = args.epochs
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        arch_id = f"{args.coupling_network}_coupling_scales{args.num_scales}_steps{args.num_steps}"
        if args.name == "debug":
            log_path = os.path.join(
                self.log_base, "debug", data_id, model_id, arch_id, optim_id, f"seed{args.seed}", time.strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            log_path = os.path.join(
                self.log_base, data_id, model_id, arch_id, optim_id, f"seed{args.seed}", args.name)

        # Move model
        model = model.to(args.device)
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)
            
        # Init parent
        super(FlowExperiment, self).__init__(model=model,
                                                   optimizer=optimizer,
                                                   scheduler_iter=scheduler_iter,
                                                   scheduler_epoch=scheduler_epoch,
                                                   log_path=log_path,
                                                   eval_every=args.eval_every,
                                                   check_every=args.check_every)
        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

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
        resume_path = os.path.join(self.log_base, self.data_id, self.model_id, self.optim_id, self.args.resume, 'check')
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
            
        super(FlowExperiment, self).run(epochs=self.args.epochs)

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
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:

            # Cast operations to mixed precision
            if self.args.super_resolution or self.args.conditional:
                batch_size = len(x[0])
                x = x[0].to(self.args.device)
                context = x[1].to(self.args.device)
                with torch.cuda.amp.autocast():
                    loss = cond_elbo_bpd(self.model, x, context)
            else:
                batch_size = len(x)
                with torch.cuda.amp.autocast():
                    loss = elbo_bpd(self.model, x.to(self.args.device))

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
            self.log_epoch("Training", loss_count, len(self.train_loader.dataset), loss_sum)

        print('')
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        
        return {'bpd': loss_sum/loss_count}

    def _train(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:
            self.optimizer.zero_grad()

            if self.args.super_resolution or self.args.conditional:
                batch_size = len(x[0])
                loss = cond_elbo_bpd(self.model, x[0].to(self.args.device), context=x[1].to(self.args.device))
            else:
                batch_size = len(x)
                loss = elbo_bpd(self.model, x.to(self.args.device))

            loss.backward()

            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()

            loss_sum += loss.detach().cpu().item() * batch_size
            loss_count += batch_size
            self.log_epoch("Training", loss_count, len(self.train_loader.dataset), loss_sum)

        print('')
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        
        return {'bpd': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in self.eval_loader:
                if self.args.super_resolution or self.args.conditional:
                    batch_size = len(x[0])
                    loss = cond_elbo_bpd(self.model, x[0].to(self.args.device), context=x[1].to(self.args.device))
                else:
                    batch_size = len(x)
                    loss = elbo_bpd(self.model, x.to(self.args.device))

                loss_sum += loss.detach().cpu().item() * batch_size
                loss_count += batch_size
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
                    epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'bpd': loss_sum/loss_count}

    def sample_fn(self, temperature=None, sample_new_batch=False):
        if self.args.samples < 1:
            return

        self.model.eval()
        get_new_batch = self.sample_batch is None or sample_new_batch
        if get_new_batch:
            self.sample_batch = next(iter(self.eval_loader))

        if self.args.super_resolution or self.args.conditional:
            imgs = self.sample_batch[0][:self.args.samples]
            context = self.sample_batch[1][:self.args.samples]
            self._cond_sample_fn(context, temperature=temperature, save_context=get_new_batch)
        else:
            imgs = self.sample_batch[:self.args.samples]
            self._sample_fn(temperature=temperature)

        if get_new_batch:
            # save real samples
            path_true_samples = '{}/samples/true_ep{}_s{}.png'.format(self.log_path, self.current_epoch, self.args.seed)
            self.save_images(imgs, path_true_samples)

    # def _sample_fn(self, checkpoint):
    def _sample_fn(self, temperature=None):
        path_samples = '{}/samples/sample_e{}_s{}.png'.format(self.log_path, self.current_epoch, self.args.seed)
        samples = self.model.sample(self.args.samples, temperature=temperature)
        self.save_images(samples, path_samples)

    # def _cond_sample_fn(self, context, checkpoint):
    def _cond_sample_fn(self, context, temperature=None, save_context=True):
        if save_context:
            # save low-resolution samples
            path_context = '{}/samples/context_e{}_s{}.png'.format(self.log_path, self.current_epoch, self.args.seed)
            self.save_images(context, path_context)

        # save samples from model conditioned on context
        path_samples = '{}/samples/sample_e{}_s{}.png'.format(self.log_path, self.current_epoch, self.args.seed)
        samples = self.model.sample(context.to(self.args.device), temperature=temperature)
        self.save_images(samples, path_samples)

    def save_images(self, imgs, file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.mkdir(os.path.dirname(file_path))
        
        out = imgs.cpu().float()
        if out.max().item() > 2:
            out /= (2**self.args.num_bits - 1)
            
        vutils.save_image(out, file_path, nrow=self.args.nrow)

    def stop_early(self, loss_dict, epoch):
        if self.args.early_stop == 0:
            return False, True

        # else check if we've passed the early stopping threshold
        current_loss = loss_dict['bpd']
        model_improved = current_loss < self.best_loss
        if model_improved:
            early_stop_flag = False
            self.best_loss = current_loss
            self.best_loss_epoch = epoch
        else:
            # model didn't improve, do we consider it converged yet?
            early_stop_count = (epoch - self.best_loss_epoch)  
            early_stop_flag = early_stop_count >= self.args.early_stop

        if early_stop_flag:
            print(f'Stopping training early: no improvement after {self.args.early_stop} epochs (last improvement at epoch {self.best_loss_epoch})')

        return early_stop_flag, model_improved


