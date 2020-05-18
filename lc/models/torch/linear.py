import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from .utils import LambdaLayer

__all__ = ['single_layer_mnist', 'single_layer_cifar10']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, input_=784):
        """
        Constructs a single-layer classifiers with linear decision boundaries (thus in linear.py).
        """
        super(LinearLayer, self).__init__()

        cfg = [('reshape', LambdaLayer(lambda x: x.view(x.size(0),-1)))]
        cfg.append(('compressible', nn.Linear(input_,10)))

        self.output = nn.Sequential(OrderedDict(cfg))
        self.loss = nn.CrossEntropyLoss()
        self.apply(_weights_init)

    def forward(self, input):
        return self.output(input)


def single_layer_mnist():
    return LinearLayer()

def single_layer_cifar10():
    inputs_ = {'MNIST': 784, 'CIFAR10': 3072, 'CIFAR10_no_mod': 3072}
    dataset = "CIFAR10_no_mod"
    return LinearLayer(input_=inputs_[dataset])