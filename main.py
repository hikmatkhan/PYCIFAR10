import argparse
import sys
import random

import numpy as np
import torch
import torchvision
from torch.nn import CrossEntropyLoss

from cifar10_models.resnet import resnet50, resnet18
from torchvision import transforms as T, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR, CyclicLR

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--outdir', type=str, default='default', help="Output results directory")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='./data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--workers', type=int, default=4, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=1024 * 4)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--max_lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--base_lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr_scheduler',  nargs="+", type=int, default=[0],
                        help="ReduceLROnPlateau(default)|ExponentialLR|MultiStepLR")
    parser.add_argument("--patience", type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=10,
                        help="The number of epoches")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
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

    net = resnet18()
    net_ref = resnet18(pretrained=True)
    print(net)

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
    optimizr = torch.optim.SGD(params=net.parameters(), lr=args.lr, momentum=args.momentum)
    # lr_schedulr = ExponentialLR(optimizer=optimizr, gamma=args.gamma)
    lr_schedulr = CyclicLR(optimizr, base_lr=args.base_lr, max_lr=args.max_lr, step_size_up=15,
                            step_size_down=25, gamma=args.gamma,
                            mode="exp_range")  # mode (str) – One of {triangular, triangular2, exp_range}
    loss_fn = CrossEntropyLoss()

    if args.gpuid[0] >= 0:
        torch.cuda.set_device(args.gpuid[0])
        net = net.cuda()
        net_ref = net_ref.cuda()
        loss_fn = loss_fn.cuda()
        gpu = True
    else:
        gpu = False

    for r in range(args.repeat):
        args.seed = r
        fix_seeds(seed=args.seed)
        for e in range(0, args.epochs):
            for i, (x, y) in enumerate(trainloader):
                # print(x.size(), " ", y.size(), " ", i)
                x, y = (x.cuda(), y.cuda()) if gpu else (x, y)
                optimizr.zero_grad()
                prd_y = net(x)
                loss = loss_fn(prd_y, y)
                loss.backward()
                optimizr.step()

                # if i % args.print_freq == 0:
                print("Repeat: ", r, " Epoch:", e, " Loss:", loss.item())
            print("*" * 100)
            print("My model accuracy:", accuracy(net, testloader, gpu))
            print("Ref model accuracy:", accuracy(net_ref, testloader, gpu))
            print("LR:", optimizr.param_groups[0]["lr"])
            print("*" * 100)
            lr_schedulr.step()

