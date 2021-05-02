import os
import pickle
import torch
import numpy as np

from survae.utils import count_parameters

from .utils import get_args_table, get_metric_table


def add_exp_args(parser):
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp', 'none'})
    parser.add_argument('--amp', type=eval, default=False, help="Use automatic mixed precision")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--teacher_model', type=str, default=None, help="Path to a pretrained teacher model")
    parser.add_argument('--cond_trans', type=str, default='quantize',
                        choices={'quantize', 'quantize4', 'quantize9', 'quantize16', 'quantize25', 'quantize36', 'split', 'split0', 'split1', 'random', 'multiply'},
                        help="Transformation type used to create low-dimensional data for super-resolution model.")

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=0.0)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=False)
    parser.add_argument('--log_wandb', type=eval, default=False)

    # Plot params
    parser.add_argument('--num_samples', type=int, default=128*1000)
    parser.add_argument('--grid_size', type=int, default=500)
    parser.add_argument('--pixels', type=int, default=1000)
    parser.add_argument('--dpi', type=int, default=96)
    parser.add_argument('--clim', type=float, default=0.05)


class BaseExperiment(object):

    def __init__(self, model, optimizer, scheduler_iter, scheduler_epoch, log_path, eval_every):

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

        # Initialize
        self.current_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []

        # automatic mixed precision
        self.amp = False
        self.scaler = None

    def train_fn(self, epoch):
        raise NotImplementedError()

    def eval_fn(self, epoch):
        raise NotImplementedError()

    def log_fn(self, epoch, train_dict, eval_dict):
        raise NotImplementedError()

    def plot_fn(self):
        raise NotImplementedError()
        
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
        print('{}. Epoch: {}/{}, Datapoint: {}/{}, NLL: {:.3f}'.format(
            title, self.current_epoch+1, self.args.epochs, loss_count, data_count, loss_sum/loss_count), end='\r')

    def create_folders(self):
        # Create log folder
        os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
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
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
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
            else:
                eval_dict = None

            # Log
            self.save_metrics()
            self.log_fn(epoch, train_dict, eval_dict)
            self.current_epoch += 1

        # Checkpoint
        self.checkpoint_save()

        # Plotting
        self.plot_fn()

    def cond_fn(self, y):
        """
        Defines how the 'high-resolution' data y should be changed into a 'low-resolution' sample.
        """
        if self.args.dataset == 'face_einstein':
            ymin = 0
            ymax = 1
        else:
            ymin = -4
            ymax = 4
            
        y = (y - ymin) / (ymax - ymin)  # set range to [0, 1]
        if self.cond_id == "quantize" or self.cond_id == "quantize4":
            x = torch.round(y)  # x is either 0 or 1, so 2D x has four possible outputs
        elif self.cond_id == "quantize9":
            x = torch.round(2 * y)  # x can be 0, 1, or 2, and a 2D x will have 9 possible outputs
        elif self.cond_id == "quantize16":
            x = torch.round(3 * y)
        elif self.cond_id == "quantize25":
            x = torch.round(4 * y)
        elif self.cond_id == "quantize36":
            x = torch.round(5 * y)
        elif self.cond_id == "split" or self.cond_id == "split0":
            x = torch.round(100.0 * y[:, :1]) / 100.0
        elif self.cond_id == "split1":
            x = torch.round(100.0 * y[:, 1:]) / 100.0
        elif self.cond_id == "multiply":
            x = torch.round(100.0 * y[:, :1] * y[:, 1:]) / 100.0
        elif self.cond_id == "random":
            r = torch.rand_like(y)
            x = y.masked_fill_(r == r.max(dim=1).values.view(y.size(0), 1), 0.0)
        else:
            raise ValueError(f"Conditional transformation {self.cond_id} is unknown.")
            
        return x

    def context_permutations(self, ymin=-10, ymax=10):
        if self.args.dataset == 'face_einstein':
            bounds = [[0, 1], [0, 1]]
        else:
            bounds = [[-4, 4], [-4, 4]]

        xv, yv = torch.meshgrid([torch.linspace(bounds[0][0], bounds[0][1], self.args.grid_size), torch.linspace(bounds[1][0], bounds[1][1], self.args.grid_size)])
        y = torch.cat([xv.reshape(-1,1), yv.reshape(-1,1)], dim=-1).to(self.args.device)
        x = self.cond_fn(y).unique(sorted=True, dim=0)

        if x.shape[0] > 10:
            x = torch.round(10.0 * x)/10.0
            x = x.unique(sorted=True, dim=0)

        print("Conditional permuations:\n", x.data)
        
        return x


                    

