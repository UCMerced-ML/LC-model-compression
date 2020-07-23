#!/bin/env/python3
from types import ModuleType
from lc.models import torch as model_def
from utils import AverageMeter, Recorder, format_time, data_loader, compute_acc_loss
import argparse
import torch
from torch import optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import os

if __name__ == '__main__':
    if not os.path.exists('references'):
        os.makedirs('references')

    model_names = model_def.__all__
    parser = argparse.ArgumentParser(description='Reference Network Trainer for MNIST and CIFAR10 networks')
    parser.add_argument('--arch', '-a', metavar='ARCH', default=model_names[0],
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: {})'.format(model_names[0]))
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--checkpoint', type=int, default=20)
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.09, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--scheduler', choices=['exponential', 'steps'], default='exponential')
    parser.add_argument('--milestones', nargs='+', default=[100,150], type=int)
    parser.add_argument('--lr_decay', type=float, default=0.99)

    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', action='store_true',
                        help='resumes from recent checkpoint')

    args = parser.parse_args()
    print(args)
    cudnn.benchmark = True
    model = getattr(model_def, args.arch)()

    model.cuda()
    train_loader, test_loader = data_loader(batch_size=args.batch_size,
                                            n_workers=args.workers,
                                            dataset=args.dataset)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=True)

    prev_state = None
    if args.resume:
        prev_state = torch.load('references/{}_checkpoint.th'.format(args.arch))

    epoch_time = AverageMeter()
    rec = Recorder()
    all_start_time = time.time()
    start_epoch = 0
    if prev_state:
        print()
        model.load_state_dict(prev_state['model_state'])
        optimizer.load_state_dict(prev_state['optimizer_state'])
        epoch_time = prev_state['epoch_time']
        rec = prev_state['records']
        all_start_time -= prev_state['training_time']
        print('Overriding provided arg with prev_state args: ', prev_state['args'])
        args = prev_state['args']
        start_epoch = prev_state['epoch']

    scheduler = None
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=start_epoch - 1)
    elif args.scheduler == 'steps':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.lr_decay, last_epoch=start_epoch-1)

    def my_eval(x, target):
        out_ = model.forward(x)
        return out_, model.loss(out_, target)

    training_time = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        model.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = x.cuda(), target.cuda()
            out = model.forward(x)
            loss = model.loss(out, target)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        epoch_time.update(end_time - start_time)
        training_time = end_time - all_start_time
        model.eval()
        print('Epoch {0} finished in {et.val:.3f}s (avg.: {et.avg:.3f}s). Training for {1}'
              .format(epoch, format_time(training_time), et=epoch_time))
        print('\tLR: {:.4}'.format(scheduler.get_lr()[0]))

        if (epoch+1)%args.print_freq == 0:
            accuracy, ave_loss = compute_acc_loss(my_eval, train_loader)
            rec.record('train', [ave_loss, accuracy, training_time, epoch+1])
            print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
            accuracy, ave_loss = compute_acc_loss(my_eval, test_loader)
            print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
            rec.record('test', [ave_loss, accuracy, training_time, epoch+1])
        scheduler.step()

        if args.checkpoint and (epoch+1) % args.checkpoint == 0:
            # create and save checkpoint here
            to_save = {'records': rec, 'epoch_time': epoch_time, 'training_time': training_time}
            to_save['model_state'] = model.state_dict()
            to_save['optimizer_state'] = optimizer.state_dict()
            to_save['lr'] = scheduler.get_lr()[0]
            to_save['epoch'] = epoch + 1
            to_save['args'] = args

            torch.save(to_save, 'references/{}_checkpoint.th'.format(args.arch))
            pass

    # training has finished. save all recorded values, parameters, end optimization states
    to_save = {'records': rec, 'epoch_time': epoch_time, 'training_time': training_time}
    to_save['model_state'] = model.state_dict()
    to_save['optimizer_state'] = optimizer.state_dict()
    to_save['lr'] = scheduler.get_lr()[0]
    to_save['epoch'] = epoch + 1
    to_save['args'] = args

    torch.save(to_save, 'references/{}.th'.format(args.arch))

    if args.checkpoint:
        # additionally remove checkpoints
        import os
        os.remove('references/{}_checkpoint.th'.format(args.arch))