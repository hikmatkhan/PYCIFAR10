import argparse
import os
import sys
import random

import numpy as np
import torch
import torchvision
from torch.nn import CrossEntropyLoss

import cifar10_models
from cifar10_models.resnet import resnet50, resnet18
from torchvision import transforms as T, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR, CyclicLR
import wandb

from utils import lr_scheduler, optimizer, log_on_wandb, init_wandb

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 4"


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0,
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model-name', type=str, default='resnet18', help="The name of actual model for the backbone")
    parser.add_argument('--outdir', type=str, default='results', help="Output results directory")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='./data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--train-aug', type=int, default=1,
                        help="Allow data augmentation during training")
    parser.add_argument('--workers', type=int, default=4, help="#Thread for dataloader")
    parser.add_argument('--batch-size', type=int, default=1024 * 10)
    parser.add_argument('--seed', type=int, default=101)

    # **********************************************************************************************
    # LR Scheduler & Optimizer Parameters
    parser.add_argument('--lr', type=float, default=0.01, help="Optimizer Learning rate")
    parser.add_argument('--max-lr', type=float, default=0.1, help="CyclicLR")
    parser.add_argument('--base-lr', type=float, default=0.0001, help="CyclicLR")
    parser.add_argument('--step-size-up', type=int, default=15, help="CyclicLR")
    parser.add_argument('--step-size-down', type=int, default=25, help="CyclicLR")
    parser.add_argument('--mode', type=str, default="exp_range", help="CyclicLR")

    parser.add_argument('--min-lr', type=float, default=0.1, help="ReduceLROnPlateau")
    parser.add_argument('--factor', type=float, default=0.9, help="ReduceLROnPlateau")
    parser.add_argument("--patience", type=float, default=0, help="Shared (ReduceLROnPlateau)")
    parser.add_argument('--metric', type=str, default="min", help="mode of the ReduceLROnPlateau")


    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.95, help="Shared")
    parser.add_argument('--lr-scheduler', type=str, default="MultiStepLR",
                        help="CyclicLR(default)|ReduceLROnPlateau|ExponentialLR|MultiStepLR")

    parser.add_argument('--weight-decay', type=float, default=0, help="Shared")
    parser.add_argument('--milestones', nargs="+", type=int, default=[20, 40, 80],
                        help="MultiStepLR")

    # **********************************************************************************************

    parser.add_argument('--epochs', type=int, default=10,
                        help="The number of epoches")
    parser.add_argument('--print-freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model-weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--eval-on-train-set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")

    # **********************************************************************************************
    # Visualization
    parser.add_argument('--wand-project', type=str, default="PYCIFAR",
                        help='Project name.')
    parser.add_argument('--username', type=str, default="hikmatkhan-",
                        help='Username')
    parser.add_argument('--wandb-log', type=int, default=0,
                        help='If True then logs will be reported on wandb.')
    # **********************************************************************************************
    args = parser.parse_args(argv)
    return args


def fix_seeds(seed=101):
    print("SEED:", seed)
    # No randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        print("Cuda detected.")
    return seed


def accuracy(net, loader, gpu):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (x, y) in loader:
            x, y = (x.cuda(), y.cuda()) if gpu else (x, y)
            # calculate outputs by running images through the network
            outputs = net(x)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #         100 * correct / total))
    return 100 * correct / total




if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    # print("H" * 100)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        os.makedirs(os.path.join(os.getcwd(), args.outdir, args.model_name))

    model = cifar10_models.resnet.__dict__[args.model_name]({})
    model_ref = cifar10_models.resnet.__dict__[args.model_name]({"pretrained": True})
    init_wandb(args, model)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    if args.train_aug:
        print("With data augmentation.")
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform

    trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False,
                                           download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    optimizr = optimizer(model=model, args=args)
    # torch.optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)
    # lr_schedulr = ExponentialLR(optimizer=optimizr, gamma=args.gamma)
    lr_schedulr = lr_scheduler(args=args, optimizer=optimizr)

    # lr_schedulr = torch.optim.lr_scheduler.__dict__[]
    # lr_schedulr = CyclicLR(optimizr, base_lr=args.base_lr, max_lr=args.max_lr, step_size_up=15,
    #                         step_size_down=25, gamma=args.gamma,
    #                         mode="exp_range")  # mode (str) – One of {triangular, triangular2, exp_range}
    loss_fn = CrossEntropyLoss()

    if args.gpuid >= 0:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()
        model_ref = model_ref.cuda()
        loss_fn = loss_fn.cuda()
        gpu = True
    else:
        gpu = False

    for r in range(args.repeat):
        args.seed = r
        fix_seeds(seed=args.seed)
        for e in range(0, args.epochs):
            avg_train_loss = 0
            for i, (x, y) in enumerate(trainloader):
                # print(x.size(), " ", y.size(), " ", i)
                x, y = (x.cuda(), y.cuda()) if gpu else (x, y)
                optimizr.zero_grad()
                prd_y = model(x)
                loss = loss_fn(prd_y, y)
                loss.backward()
                optimizr.step()

                # if i % args.print_freq == 0:
                print("I:", i, " Repeat: ", r, " Epoch:", e, " Loss:", loss.item())
                avg_train_loss += loss.detach().item()
                log_on_wandb(args, {"train_b_loss": loss.item()})
            # break
            avg_train_loss = avg_train_loss / (i + 1)
            print("*" * 100)
            print("avg_train_loss=", avg_train_loss)
            print("My model accuracy:", accuracy(model, testloader, gpu))
            log_on_wandb(args, {"test_accuracy": accuracy(model, testloader, gpu)})
            print("Ref model accuracy:", accuracy(model_ref, testloader, gpu))
            log_on_wandb(args, {"ref_test_accuracy": accuracy(model_ref, testloader, gpu)})
            print("LR:", optimizr.param_groups[0]["lr"])
            log_on_wandb(args, {"lr": optimizr.param_groups[0]["lr"]})
            log_on_wandb(args, {"avg_train_loss": avg_train_loss})
            print("*" * 100)
            if isinstance(lr_schedulr, torch.optim.lr_scheduler.ReduceLROnPlateau):
                print("ReduceLROnPlateau")
                lr_schedulr.step(avg_train_loss)
            else:
                lr_schedulr.step()

