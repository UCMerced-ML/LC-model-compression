import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from .utils import LambdaLayer

__all__ = ['lenet300_classic', 'lenet300_classic_drop', 'lenet300_modern', 'lenet300_modern_drop']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class LeNet300(nn.Module):
    """
    Base LeNet300 module that allows to reconfigure it. LeNet300 is a network having 3 layers with weights of
    dimensions 784x300, 300x100 and 100x10 and trained on MNIST dataset.

    References:
        Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner
        Gradient-based learning applied to document recognition
        https://ieeexplore.ieee.org/document/726791
    """
    def __init__(self, dropout, nonlinearity):
        """
        Constructor of LeNet300. With given options it is possible to set various nonlinearities and
        add dropout layers after them
        :param dropout: boolean, if True, the dropout layer will be added after every inter hidden layer,
            no dropout after last layer
        :param nonlinearity: function, a constructor that returns nonlinearity function
        """
        super(LeNet300, self).__init__()

        layers = [(784, 300), (300, 100), (100, 10)]

        cfg = [('reshape', LambdaLayer(lambda x: x.view(x.size(0),-1)))]
        for i, l in enumerate(layers):
            cfg.append(('compressible_' + str(i), nn.Linear(*l)))
            if i != len(layers)-1:
                # only non terminal layers have nonlinearity and (possible) dropouts
                cfg.append(('nonlinearity_' + str(i), nonlinearity()))
                if dropout:
                    cfg.append(('drop_'+str(i), nn.Dropout()))

        self.output = nn.Sequential(OrderedDict(cfg))
        self.loss = nn.CrossEntropyLoss()
        self.apply(_weights_init)

    def forward(self, input):
        return self.output(input)


def lenet300_classic():
    """
    Creates classical version of LeNet300, the one having tanh activation functions and
    no dropouts
    """
    return LeNet300(dropout=False, nonlinearity=nn.Tanh)


def lenet300_classic_drop():
    """
    Returns classical LeNet300 with intermediate dropouts between layers.
    """
    return LeNet300(dropout=True, nonlinearity=nn.Tanh)


def lenet300_modern():
    return LeNet300(dropout=False, nonlinearity=nn.ReLU)


def lenet300_modern_drop():
    return LeNet300(dropout=True, nonlinearity=nn.ReLU)