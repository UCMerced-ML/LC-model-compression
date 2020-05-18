'''
Modified from https://github.com/pytorch/vision.git
    and from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py

'''
import math
import torch.nn as nn
from collections import OrderedDict
__all__ = [
    'VGGcif', 'vggcif11', 'vggcif11_bn', 'vggcif13', 'vggcif13_bn', 'vggcif16', 'vggcif16_bn',
    'vggcif19_bn', 'vggcif19',
]

from .utils import LambdaLayer, weight_decay
class VGGcif(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, old=True):
        super(VGGcif, self).__init__()
        #self.features = features
        self.output = nn.Sequential(OrderedDict([
            ('features', features),
            ('reshape', LambdaLayer(lambda x: x.view(x.size(0),-1))),
            ('drop_classifier_1', nn.Dropout()),
            ('compressible_classifier_1', nn.Linear(512, 512)),
            ('nonlineariy_classifier_1', nn.ReLU(True)),
            ('drop_classifier_2', nn.Dropout()),
            ('compressible_classifier_2', nn.Linear(512, 512)),
            ('nonlinearity_classifier_2', nn.ReLU(True)),
            ('compressible_classifier_2', nn.Linear(512, 10))])
        )

        self.weight_decay = weight_decay(self, old=old)
        self.loss = lambda x, target: nn.CrossEntropyLoss()(x,target) + 5e-4*self.weight_decay()
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
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [('maxpool_features_' + str(i), nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [('compressible_conv_'+str(i), conv2d),
                           ('bn_conv_'+str(i), nn.BatchNorm2d(v)),
                           ('nonlinearity_conv_'+str(i),nn.ReLU(inplace=True))]
            else:
                layers += [('compressible_conv_'+str(i), conv2d),
                           ('nonlinearity_conv_'+str(i),nn.ReLU(inplace=True))]
            in_channels = v
    return nn.Sequential(OrderedDict(layers))


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '9': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

def vggcif9():
    return VGGcif(make_layers(cfg['A']))

def vggcif9_bn():
    return VGGcif(make_layers(cfg['A'], batch_norm=True))

def vggcif11():
    """VGG 11-layer model (configuration "A")"""
    return VGGcif(make_layers(cfg['A']))


def vggcif11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGGcif(make_layers(cfg['A'], batch_norm=True))


def vggcif13():
    """VGG 13-layer model (configuration "B")"""
    return VGGcif(make_layers(cfg['B']))


def vggcif13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGGcif(make_layers(cfg['B'], batch_norm=True))


def vggcif16():
    """VGG 16-layer model (configuration "D")"""
    return VGGcif(make_layers(cfg['D']))


def vggcif16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGGcif(make_layers(cfg['D'], batch_norm=True))


def vggcif19():
    """VGG 19-layer model (configuration "E")"""
    return VGGcif(make_layers(cfg['E']))


def vggcif19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    module = VGGcif(make_layers(cfg['E'], batch_norm=True))
    return module