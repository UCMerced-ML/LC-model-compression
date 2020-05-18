import torch.nn as nn
from torch.nn.init import xavier_uniform_
from collections import OrderedDict
from .utils import LambdaLayer
__all__ = ['lenet5_classic', 'lenet5_drop']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class LeNet5(nn.Module):
    """
    LeNet5 as implemented in Caffe/Caffe2.

    At the time of writing this code, two different versions of LeNet5 network is coexisting. The original one due
    to LeCun et al. have 60k parameters; the other one, first appeared in Caffe framework [2] has 266k parameters and
    vaguely similar to LeNet5 network defined in [1]. Although authors of Caffe did not call this network LeNet5
    explicitly (simply referring to it as LeNet), the community knows this model as LeNet5-Caffe and for some others
    it is the only version of LeNet5.

    References:
    1.  Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner
        Gradient-based learning applied to document recognition
        https://ieeexplore.ieee.org/document/726791

    2.  Implementation details of LeNet5 network for Caffe
        https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
    """
    def __init__(self, dropout, nonlinearity):
        super(LeNet5, self).__init__()
        self.special = True
        filters = [(20, 5), (50, 5)]
        layers = [(800, 500), (500, 10)]

        cfg = []
        cfg.append(['init_reshape', LambdaLayer(lambda x: x.view(x.size(0), 1,28,28))])
        for i, f in enumerate(filters):
            prev = 1 if i==0 else filters[i-1][0]
            cfg.append(('compressible_' + str(i), nn.Conv2d(prev, f[0], f[1])))
            cfg.append(('nonlineairy_'+str(i), nonlinearity()))
            cfg.append(('maxpool_'+str(i), nn.MaxPool2d(kernel_size=(2,2), stride=2)))


        cfg.append(['reshape', LambdaLayer(lambda x: x.view(x.size(0),-1))])
        for i, l in enumerate(layers):
            cfg.append(('compressible_' + str(i+len(filters)), nn.Linear(*l)))
            if i != len(layers)-1:
                # only non terminal layers have nonlinearity and (possible) dropouts
                cfg.append(('nonlinearity_' + str(i+len(filters)), nonlinearity()))
                if dropout:
                    cfg.append(('drop_'+str(i+len(filters)), nn.Dropout()))

        print(cfg)

        self.output = nn.Sequential(OrderedDict(cfg))
        self.loss = nn.CrossEntropyLoss()
        self.apply(_weights_init)

    def forward(self, input):
        return self.output(input)

def lenet5_classic():
    return LeNet5(dropout=False, nonlinearity=lambda: nn.ReLU(True))


def lenet5_drop():
    return LeNet5(dropout=True, nonlinearity=lambda: nn.ReLU(True))