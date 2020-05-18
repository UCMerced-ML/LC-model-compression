import torch
from torch import nn
import numpy as np


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def count_params(module, compressible=False):
    assert isinstance(module, nn.Module)
    if not compressible:
        return sum(map(lambda x: np.prod(x.data.numpy().shape),
                       filter(lambda p: p.requires_grad, module.parameters())))

    return sum(map(lambda x: np.prod(x[1].weight.data.numpy().shape),
                   filter(lambda p: 'compressible' in p[0], module.named_modules())))


def weight_decay(module):
    decayed = []
    for name, value in module.named_parameters():
        if name.endswith('.weight') and value.requires_grad:
            decayed.append(value)

    def wd():
        if len(decayed) > 0:
            sum_ = torch.tensor(0.0, requires_grad=True).to(dtype=decayed[0].dtype, device=decayed[0].device)
            for x in decayed:
                sum_ += torch.sum(x**2)
            return sum_

        return 0
    return wd


def weight_decay_layers_only(module):
    decayed = []
    for sub_module in module.modules():
        print(sub_module)
        if isinstance(sub_module, nn.Linear) or \
                isinstance(sub_module, nn.Conv2d):
            decayed.append(sub_module.weight)
    for p in decayed:
        print(type(p), p.shape)

    def wd():
        sum_ = torch.tensor(0.0, requires_grad=True).to(dtype=decayed[0].dtype, device=decayed[0].device)
        for val in decayed:
            sum_ += torch.sum(val ** 2)
        return sum_

    return wd
