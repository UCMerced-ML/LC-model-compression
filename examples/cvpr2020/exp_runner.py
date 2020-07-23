import sys,os
sys.path.append(os.getcwd())

import cifar10_exp_setups
from modified_lc import RankSelectionLcAlg as LCAlgorithm
import argparse
import time
from torch.backends import cudnn
cudnn.benchmark = True



def lc_exp_runner(exp_setup, lc_config, l_step_config, c_step_config, resume=False):
    if not os.path.exists('results'):
        os.makedirs('results')

    l_step_optimization, evaluation, create_lc_compression_task = exp_setup.lc_setup()
    compression_tasks = create_lc_compression_task(c_step_config)

    lc_alg = LCAlgorithm(model=exp_setup.model, compression_tasks=compression_tasks,
                         lc_config=lc_config, l_step_optimization=l_step_optimization,
                         evaluation_func=evaluation, l_step_config=l_step_config)
    if resume:
        lc_alg.run(name=exp_setup.name, tag=lc_config['tag'], restore=True)
    else:
        lc_alg.run(name=exp_setup.name, tag=lc_config['tag'])


def ft_exp_runner(exp_setup, ft_config, c_step_config):
    finetuning_func = exp_setup.finetune_setup(tag_of_lc_model=ft_config['tag'], c_step_config=c_step_config)
    finetuning_func(ft_config)

if __name__ == "__main__":
    if not os.path.exists('references'):
        raise Exception("You don't have any trained reference networks to compress. "
                        "Please use `reference_trainer.py' to train a reference model.")

    parser = argparse.ArgumentParser(description='PyTorch LC RankSelection Training')

    #--------------------
    parser.add_argument('--exp_setup', choices=cifar10_exp_setups.__all__, default='None')
    parser.add_argument('--type', choices=['lc', 'ft'], default='lc')
    parser.add_argument('--tag', type=str, default="tag")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--continue_with_original_config', action='store_true')
    # -------------------
    # lc-config
    parser.add_argument('--lc_steps', type=int, default=20)
    parser.add_argument('--mu_init', type=float, default=9e-5)
    parser.add_argument('--mu_inc', type=float, default=1.09)
    parser.add_argument('--mu_rep', type=int, default=1)
    # c-step-config
    parser.add_argument('--conv_scheme', choices=['scheme_1', 'scheme_2'], default='scheme_1')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--criterion', choices=['storage', 'flops'], default='storage')
    # l-step-config
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.09, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', '--learning-rate-decay', default=0.98, type=float, metavar='LRD', help='learning rate decay')
    parser.add_argument('--lr_decay_mode', choices=['after_l', 'restart_on_l'], default='after_l')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', '-p', default=5, type=int, metavar='N', help='print frequency (default: 5)')
    parser.add_argument('--name', default='')
    # imagenet experiments
    parser.add_argument('--full-sized', dest='full_sized', action='store_true',
                        help='use full-sized images or resized (shortest side with 256) as main source of data')
    parser.add_argument('--data-dir', default='', type=str, metavar='PATH',
                        help='path for lmdb stores for data')
    parser.add_argument('--batch-from-disk', type=int, default=150,
                        help='controls in what sequences we read data from lmdb, shorter sequences impact reading speed.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--val-batch-size', default=500, type=int,
                        metavar='N',
                        help='mini-batch size (default: 500), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    # -------------------
    args = parser.parse_args()
    # -------8<-----
    if args.type == 'lc':
        exp_setup = getattr(cifar10_exp_setups, args.exp_setup)()

        l_step_config = {
            'lr_decay_mode': args.lr_decay_mode,
            'lr': args.lr,
            'epochs': args.epochs,
            'momentum': args.momentum,
            'print_freq': args.print_freq,
            'lr_decay': args.lr_decay,
            'imagenet_config': {
                "full_sized": args.full_sized,
                "data_dir": args.data_dir,
                "batch_from_disk": args.batch_from_disk,
                "workers": args.workers,
                "half": args.half,
                "batch_size": args.batch_size,
                "val_batch_size": args.val_batch_size
            }
        }
        c_step_config = {
            'alpha': args.alpha,
            'criterion': args.criterion,
            'conv_scheme': args.conv_scheme
        }
        lc_config = {
            'mu_init': args.mu_init,
            'mu_inc': args.mu_inc,
            'mu_rep': args.mu_rep,
            'steps': args.lc_steps,
            'tag': args.tag
        }
        exp_setup.eval_config=l_step_config
        lc_exp_runner(exp_setup, lc_config, l_step_config, c_step_config, resume=args.resume)

    elif args.type == 'ft':
        #finetuning
        exp_setup = getattr(cifar10_exp_setups, args.exp_setup)()

        c_step_config = {
            'alpha': args.alpha,
            'criterion': args.criterion,
            'conv_scheme': args.conv_scheme
        }

        ft_config = {
            'lr': args.lr,
            'epochs': args.epochs,
            'momentum': args.momentum,
            'print_freq': args.print_freq,
            'lr_decay': args.lr_decay,
            'tag': args.tag,
            'imagenet_config': {
                "full_sized": args.full_sized,
                "data_dir": args.data_dir,
                "batch_from_disk": args.batch_from_disk,
                "workers": args.workers,
                "half": args.half,
                "batch_size": args.batch_size,
                "val_batch_size": args.val_batch_size
            }
        }
        ft_exp_runner(exp_setup, ft_config, c_step_config)