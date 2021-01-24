import torch
import torchvision.utils as vutils

from survae.distributions import DataParallelDistribution
from survae.utils import elbo_bpd
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
    #parser.add_argument('--amp', type=eval, default=False, help="Use automatic mixed precision")
    parser.add_argument('--resume', type=str, default=None)

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=0.0)
    parser.add_argument('--annealing_schedule', type=int, default=0, help="Number of epochs to anneal NDP loss")
    parser.add_argument('--early_stop', type=int, default=0, help="Number of epochs with no improvement before stopping training")

    # sampling parameters
    parser.add_argument('--save_samples', type=eval, default=True, help="Save image samples after training")
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

    log_base = os.path.join(get_survae_path(), 'experiments/manifold/log')
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

        if args.name == "debug":
            log_path = os.path.join(
                self.log_base, "debug", data_id, model_id, optim_id, f"seed{args.seed}", time.strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            log_path = os.path.join(
                self.log_base, data_id, model_id, optim_id, f"seed{args.seed}", args.name)

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
                                             check_every=args.check_every,
                                             save_samples=args.save_samples)

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
        #self.scaler = torch.cuda.amp.GradScaler() if args.amp and args.parallel != 'dp' else None
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

    def beta_annealing(self, epoch, min_beta=0.0, max_beta=1.0):
        """
        Beta annealing term to control the weight of the NDP portion of the flow.
        Often it can help to let the bijective portion of the flow begin to map to
        a the first base distribution before expecting the NDP portion of the flow find a 
        low dimensional representation.
        Only applicable when using 2 (or more) base distributions
        """
        if self.args.annealing_schedule > 0 and self.args.compression == "vae" and len(self.args.base_distributions) > 1:
            return max(min([(epoch * 1.0) / max([self.args.annealing_schedule, 1.0]), max_beta]), min_beta)
        else:
            return None

    def run(self):
        if self.args.resume:
            self.resume()
            
        super(FlowExperiment, self).run(epochs=self.args.epochs)

    def train_fn(self, epoch):
        if self.scaler is not None:
           # use automatic mixed precision
           return self._train_amp(epoch)
        else:
            return self._train(epoch)

    def _train_amp(self, epoch):
        """
        Same training procedure, but uses half precision to speed up training on GPUs

        NOTE: Not currently implemented, this only runs on the latest pytorch versions
              so I'm leaving this out for now.
        """
        self.model.train()
        beta = self.beta_annealing(epoch)
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:
            self.optimizer.zero_grad()

            # Cast operations to mixed precision
            with torch.cuda.amp.autocast():
                loss = elbo_bpd(self.model, x.to(self.args.device), beta=beta)

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

            # accumulate loss and report
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
                epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')

        print('')
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        
        return {'bpd': loss_sum/loss_count}

    def _train(self, epoch):
        self.model.train()
        beta = self.beta_annealing(epoch)
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:
            self.optimizer.zero_grad()

            loss = elbo_bpd(self.model, x.to(self.args.device), beta=beta)
            loss.backward()

            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()

            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
                epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')

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
                loss = elbo_bpd(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(
                    epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'bpd': loss_sum/loss_count}

    def sample_fn(self):
        self.model.eval()

        path_check = '{}/check/checkpoint.pt'.format(self.log_path)
        checkpoint = torch.load(path_check)

        path_samples = '{}/samples/sample_ep{}_s{}.png'.format(
            self.log_path, checkpoint['current_epoch'], self.args.seed)
        if not os.path.exists(os.path.dirname(path_samples)):
            os.mkdir(os.path.dirname(path_samples))

        # save model samples
        samples = self.model.sample(self.args.samples).cpu().float() / (2**self.args.num_bits - 1)
        vutils.save_image(samples, path_samples, nrow=self.args.nrow)
                
        # save real samples too
        path_true_samples = '{}/samples/true_ep{}_s{}.png'.format(self.log_path, checkpoint['current_epoch'], self.args.seed)
        imgs = next(iter(self.eval_loader))[:self.args.samples]
        vutils.save_image(imgs.cpu().float(), path_true_samples, nrow=self.args.nrow)

    def stop_early(self, loss_dict, epoch):
        if self.args.early_stop == 0 or epoch < self.args.annealing_schedule:
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
            early_stop_flag = early_stop_count > self.args.early_stop

        if early_stop_flag:
            print(f'Stopping training early: no improvement after {self.args.early_stop} epochs (last improvement at epoch {self.best_loss_epoch})')

        return early_stop_flag, model_improved


