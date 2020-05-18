'''
Modified from https://github.com/pytorch/vision.git
    and from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py

'''
import math
import torch.nn as nn
from collections import OrderedDict
__all__ = [ 'nincif_bn' ]

from .utils import LambdaLayer, weight_decay_layers_only
class NINcif(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(NINcif, self).__init__()

        self.output = nn.Sequential(OrderedDict([
            ('features', features),
            ('reshape', LambdaLayer(lambda x: x.view(x.size(0), 10)))
            ])
        )
        self.old_weight_decay=False
        self.weight_decay = weight_decay_layers_only(self)
        self.loss = lambda x, target: nn.CrossEntropyLoss()(x,target) + 0.5e-4*self.weight_decay()

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, input):
        return self.output(input)


def make_layers(cfg, batch_norm=False):
    layers = []

    for i, v in enumerate(cfg):
        if v[0] == 'M':
            _, k, s, p = v
            layers += [('maxpool_features_' + str(i), nn.MaxPool2d(kernel_size=k, stride=s, padding=p))]
            if i!=len(cfg)-1:
                layers += [('dropout_' + str(i), nn.Dropout(inplace=True))]
            else:
                print('BLA bla bla', layers.pop(-2))
        else:
            _, in_, out_, k, s, p = v
            conv2d = nn.Conv2d(in_, out_, kernel_size=k, stride=s, padding=p)
            if batch_norm:
                layers += [('compressible_conv_'+str(i), conv2d),
                           ('bn_conv_'+str(i), nn.BatchNorm2d(out_)),
                           ('nonlinearity_conv_'+str(i),nn.ReLU(inplace=True))]
            else:
                layers += [('compressible_conv_'+str(i), conv2d),
                           ('nonlinearity_conv_'+str(i),nn.ReLU(inplace=True))]
    print(layers)
    return nn.Sequential(OrderedDict(layers))


cfg = [('C',   3, 192, 5, 1, 2),
       ('C', 192, 160, 1, 1, 0),
       ('C', 160,  96, 1, 1, 0),
       ('M', 3, 2, 1),
       ('C',  96, 192, 5, 1, 2),
       ('C', 192, 192, 1, 1, 0),
       ('C', 192, 192, 1, 1, 0),
       ('M', 3, 2, 1),
       ('C', 192, 192, 3, 1, 1),
       ('C', 192, 192, 1, 1, 0),
       ('C', 192,  10, 1, 1, 0),
       ('M', 8, 1, 0),
       ]

def nincif_bn():
    return NINcif(make_layers(cfg, batch_norm=True))

def nincif_init():
    return NINcif(make_layers(cfg, batch_norm=False))
