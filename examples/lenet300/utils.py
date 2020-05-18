import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
import os
from torch.utils.data import TensorDataset, DataLoader


def mnist_data():
    if os.path.isfile('./datasets/mnist_dataset.pkl'):
        return torch.load('./datasets/mnist_dataset.pkl')
    else:
        mnist_train = torchvision.datasets.MNIST(root='./datasets/', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(root='./datasets/', train=False, download=True)

        train_data = mnist_train.data.to(torch.float) / 255.
        test_data = mnist_test.data.to(torch.float) / 255.
        mean_image = torch.mean(train_data, dim=0)

        train_data -= mean_image
        test_data -= mean_image

        train_labels = mnist_train.targets
        test_labels = mnist_test.targets

        our_mnist = {
            'train_data': train_data, 'test_data': test_data,
            'train_labels': train_labels, 'test_labels': test_labels
        }
        torch.save(our_mnist, './datasets/mnist_dataset.pkl')
        return our_mnist


def data_loader(batch_size=256, dataset='',n_workers=0):
    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=False)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=False)

        return train_loader, val_loader
    elif dataset=='CIFAR10_no_mod':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=False)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=False)

        return train_loader, val_loader
    elif dataset =='MNIST':
        data = mnist_data()
        print(data['train_data'].size())
        train_data = TensorDataset(data['train_data'], data['train_labels'])
        test_data = TensorDataset(data['test_data'], data['test_labels'])
        print(data['test_data'].size())
        train_loader = DataLoader(train_data, num_workers=n_workers, batch_size=batch_size, shuffle=True,
                                  pin_memory=True)
        test_loader = DataLoader(test_data, num_workers=n_workers, batch_size=batch_size, shuffle=False,
                                 pin_memory=True)

        return train_loader, test_loader


def format_time(seconds):
    if seconds < 60:
        return '{:.1f}s.'.format(seconds)
    if seconds < 3600:
        return '{:d}m. {}'.format(int(seconds//60), format_time(seconds%60))
    if seconds < 3600*24:
        return '{:d}h. {}'.format(int(seconds//3600), format_time(seconds%3600))

    return '{:d}d. {}'.format(int(seconds//(3600*24)), format_time(seconds%(3600*24)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder():
    def __init__(self):
        pass

    def record(self, tag, value):
        if hasattr(self, tag):
            self.__dict__[tag].append(value)
        else:
            self.__dict__[tag] = [value]


def compute_acc_loss(forward_func, data_loader):
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(data_loader):
        with torch.no_grad():
            target = target.cuda()
            score, loss = forward_func(x.cuda(), target)
            _, pred_label = torch.max(score.data, 1)
            correct_cnt += (pred_label == target.data).sum().item()
            ave_loss += loss.data.item() * len(x)
    accuracy = correct_cnt * 1.0 / len(data_loader.dataset)
    ave_loss /= len(data_loader.dataset)
    return accuracy, ave_loss