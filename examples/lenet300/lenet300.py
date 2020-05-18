import lc
from lc.torch import ParameterTorch as Param, AsVector, AsIs
from lc.compression_types import ConstraintL0Pruning, LowRank, RankSelection, AdaptiveQuantization
from lc.models.torch import lenet300_classic
from utils import compute_acc_loss

import argparse
import gzip
import pickle
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets


def data_loader(batch_size=256, n_workers=4):
    train_data_th = datasets.MNIST(root='./datasets', download=True, train=True)
    test_data_th = datasets.MNIST(root='./datasets', download=True, train=False)

    data_train = np.array(train_data_th.train_data[:]).reshape([-1, 28 * 28]).astype(np.float32)
    data_test = np.array(test_data_th.test_data[:]).reshape([-1, 28 * 28]).astype(np.float32)
    data_train = (data_train / 255)
    dtrain_mean = data_train.mean(axis=0)
    data_train -= dtrain_mean
    data_test = (data_test / 255).astype(np.float32)
    data_test -= dtrain_mean

    train_data = TensorDataset(torch.from_numpy(data_train), train_data_th.train_labels)
    test_data = TensorDataset(torch.from_numpy(data_test), test_data_th.test_labels)

    train_loader = DataLoader(train_data, num_workers=n_workers, batch_size=batch_size, shuffle=True,)
    test_loader = DataLoader(test_data, num_workers=n_workers, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_reference():
    pass

def main(exp_name="pruning"):
    device = torch.device('cuda')
    net = lenet300_classic().to(device)

    # loading the parameters of pre-trained lenet300
    try:
        with gzip.open('lenet300_classic.pklz', 'rb') as ff:
            state_dict = pickle.load(ff)
    except FileNotFoundError:
        # trained reference is missing
        state_dict = train_reference()

    net.load_state_dict(state_dict)
    train_loader, test_loader = data_loader(256)

    # check the loaded network results
    net.eval()
    def forward_func(x, target):
        y=net(x)
        return y, net.loss(y, target)
    accuracy, ave_loss = compute_acc_loss(forward_func, train_loader)
    print('==>>> Loaded train loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
    accuracy, ave_loss = compute_acc_loss(forward_func, test_loader)
    print('==>>> Loaded test loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))

    mu_s = None
    compression_tasks = None
    lr_base = None
    epochs_per_step = 20

    # Notice the only things to be changed are:
    #   1) compression settings, i.e., the parameters to be compressed
    #   2) applied compression types, e.g., quantization or pruning
    if exp_name == 'pruning':
        # example settings for pruning, which would achieve 5% non-zero weights
        selected_modules = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]
        compression_tasks = {
            Param(selected_modules, device): (AsVector, ConstraintL0Pruning(kappa=13310), 'pruning')
        }
        mu_s = [9e-5 * (1.1 ** n) for n in range(40)]
        lr_base = 0.1

    elif exp_name == "quantize_all":
        # example settings for quantization, where every layer is quantized with k=2 separate codebook
        compression_tasks = {}
        i=0
        for w in [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]:
            compression_tasks[Param(w, device)] = (AsVector, AdaptiveQuantization(k=2), f'task_{i}')
            i+=1

        mu_s = [9e-5 * (1.1 ** n) for n in range(40)]
        lr_base = 0.09

    elif exp_name == "quantize_two_layers":
        # example settings for quantization, where first and last layer is quantized with k=2 separate codebook
        compression_tasks = {}
        i=0
        for w in [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]:
            if i != 1:
                compression_tasks[Param(w, device)] = (AsVector, AdaptiveQuantization(k=2), f'task_{i}')
            i += 1

        mu_s = [9e-5 * (1.1 ** n) for n in range(40)]
        lr_base = 0.09

    elif exp_name == "all_mixed":
        # example settings  where the first is pruned with 5000 remaining values, the second layer is compressed with a
        # low-rank matrix and the third layer is quantized with k=2 codebook,
        compression_tasks = {}
        i=0
        for w in [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]:
            if i == 0:
                compression_tasks[Param(w, device)] = (AsVector, ConstraintL0Pruning(kappa=5000), 'pruning')
            if i == 1:
                compression_tasks[Param(w, device)] = (AsIs, LowRank(target_rank=10, conv_scheme=None), 'low-rank')
            if i == 2:
                compression_tasks[Param(w, device)] = (AsVector, AdaptiveQuantization(k=2), 'quantization')
            i += 1

        mu_s = [9e-5 * (1.4 ** n) for n in range(40)]
        lr_base = 0.05

    elif exp_name == "low_rank":
        # example setting for low rank where the weights matrices are constrained to specific ranks
        compression_tasks = {}
        ranks = [80, 7, 9]
        i=0
        for w, r in zip([lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)], ranks):
            compression_tasks[Param(w, device)] = (AsIs, LowRank(target_rank=r, conv_scheme=None), f'task_{i}')
            i+=1

        mu_s = [1e-3 * (1.3 ** n) for n in range(40)]
        lr_base = 0.01

    elif exp_name == 'low_rank_with_selection':
        compression_tasks = {}
        alpha=1e-6
        for i, (w, module) in enumerate([((lambda x=x: getattr(x, 'weight')), x) for x in net.modules() if isinstance(x, nn.Linear)]):
            compression_tasks[Param(w, device)] \
                = (AsIs, RankSelection(conv_scheme='scheme_1', alpha=alpha, criterion='storage', normalize=False,
                                       module=module), f"task_{i}")
        mu_s = [1e-3 * (1.1 ** n) for n in range(40)]
        lr_base = 0.1

    elif exp_name == 'additive_quant_and_prune':
        selected_modules = [lambda x=x: getattr(x, 'weight') for x in net.modules() if isinstance(x, nn.Linear)]
        compression_tasks = {
            Param(selected_modules, device): [
                (AsVector, ConstraintL0Pruning(kappa=2662), 'pruning'),
                (AsVector, AdaptiveQuantization(k=2), 'quant')
            ]
        }
        mu_s = [9e-5 * (1.1 ** n) for n in range(40)]
        lr_base = 0.09

    def train_test_acc_eval_f(model):
        def forward_func(x, target):
            y = net(x)
            return y, net.loss(y, target)
        acc_train, loss_train = compute_acc_loss(forward_func, train_loader)
        acc_test, loss_test = compute_acc_loss(forward_func, test_loader)

        print(f"Train acc: {acc_train*100:.2f}%, train loss: {loss_train}")
        print(f"TEST ACC: {acc_test*100:.2f}%, test loss: {loss_test}")

    def my_l_step(model, lc_penalty, step):
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        lr = lr_base*(0.98**step)
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
        print(f'L-step #{step} with lr: {lr:.5f}')
        epochs_per_step_ = epochs_per_step
        if step == 0:
            epochs_per_step_ = epochs_per_step_ * 2
        for epoch in range(epochs_per_step_):
            avg_loss = []
            for x, target in train_loader:
                optimizer.zero_grad()
                x = x.to(device)
                target = target.to(dtype=torch.long, device=device)
                out = model(x)
                loss = model.loss(out, target) + lc_penalty()
                avg_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            print(f"\tepoch #{epoch} is finished.")
            print(f"\t  avg. train loss: {np.mean(avg_loss):.6f}")

    lc_alg = lc.Algorithm(
        model=net,                            # model to compress
        compression_tasks=compression_tasks,  # specifications of compression
        l_step_optimization=my_l_step,        # implementation of L-step
        mu_schedule=mu_s,                     # schedule of mu values
        evaluation_func=train_test_acc_eval_f # evaluation function
    )
    lc_alg.run()                              # entry point to the LC algorithm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LC compression of LeNet300 with various combinations')
    parser.add_argument('--exp_name', choices=["pruning", "quantize_all", "quantize_two_layers", "all_mixed",
                                               "low_rank", "low_rank_with_selection", "additive_quant_and_prune"],
                        default='lc')
    args = parser.parse_args()

    main(args.exp_name)