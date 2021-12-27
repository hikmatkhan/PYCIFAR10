import torch
import wandb


def lr_scheduler(args, optimizer):
    lr_schedular_arg = {"optimizer": optimizer}
    if args.lr_scheduler in ["ExponentialLR"]:
        lr_schedular_arg["gamma"] = args.gamma
    elif args.lr_scheduler in ["CyclicLR"]:
        lr_schedular_arg["base_lr"] = args.base_lr
        lr_schedular_arg["max_lr"] = args.max_lr
        lr_schedular_arg["step_size_up"] = args.step_size_up
        lr_schedular_arg["step_size_down"] = args.step_size_down
        lr_schedular_arg["gamma"] = args.gamma
        lr_schedular_arg["mode"] = args.mode
    elif args.lr_scheduler in ["MultiStepLR"]:
        lr_schedular_arg["milestones"] = args.milestones
        lr_schedular_arg["gamma"] = args.gamma
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_schedular_arg["mode"] = args.metric
    else:
        print("LR Scheduler didn't select.")
        raise AssertionError()
    print("Lr_Scheduler:", args.lr_scheduler)
    # from torch.optim.lr_scheduler
    return torch.optim.lr_scheduler.__dict__[args.lr_scheduler](**lr_schedular_arg)


def optimizer(model, args):
    optimizer_arg = {'params': model.parameters(),
                     'lr': args.lr,
                     'weight_decay': args.weight_decay}
    if args.optimizer in ['SGD', 'RMSprop']:
        optimizer_arg['momentum'] = args.momentum
    elif args.optimizer in ['Rprop']:
        optimizer_arg.pop('weight_decay')
    elif args.optimizer == 'amsgrad':
        optimizer_arg['amsgrad'] = True
        args.optimizer = 'Adam'
    elif args.optimizer in ["Adam"]:
        optimizer_arg["lr"] = args.lr
    else:
        print("Optimizer didn't select.")
        raise AssertionError()
    return torch.optim.__dict__[args.optimizer](**optimizer_arg)


def init_wandb(args, model=None):
    wandb.init(project=args.wand_project, entity=args.username, reinit=True)
    if model is not None:
        wandb.watch(model, log_freq=10)
