import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from survae.optim.schedulers import LinearWarmupScheduler


optim_choices = {'sgd', 'adam', 'adamax'}


def add_optim_args(parser):

    # Model params
    parser.add_argument('--optimizer', type=str, default='adamax', choices=optim_choices)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--momentum_sqr', type=float, default=0.999)
    parser.add_argument('--exponential_lr', type=eval, default=True)



def get_optim_id(args):
    lr_str = f"_lr{args.lr:.0e}"
    exp_str = '_expdecay' if args.exponential_lr else ''
    warmup_str = f"_warmup{args.warmup}" if args.warmup is not None else ''
    return f"{args.optimizer}{lr_str}{exp_str}{warmup_str}"


def get_optim(args, model):
    assert args.optimizer in optim_choices

    if args.boosted_components > 1:
        flow_params = {f"flows.{c}": torch.nn.ParameterList() for c in range(args.boosted_components)}
        for name, param in model.named_parameters():
            pos1 = name.find(".") + 1
            pos2 = name[(pos1):].find(".") + pos1
            component = name[:pos2]
            flow_params[component].append(param)
        
        params = [{'params': flow_params[f"flows.{c}"]} for c in range(args.boosted_components)]
    else:
        params = model.parameters()        

    # Base optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(params, lr=args.lr, betas=(args.momentum, args.momentum_sqr))

    # warmup LR
    if args.warmup is not None and args.warmup > 0:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else:
        scheduler_iter = None

    # Exponentially decay LR
    if args.exponential_lr:
        scheduler_epoch = ExponentialLR(optimizer, gamma=0.995)
    else:
        scheduler_epoch = None
    

    return optimizer, scheduler_iter, scheduler_epoch

