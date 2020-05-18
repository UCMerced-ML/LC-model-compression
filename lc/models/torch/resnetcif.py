'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict
from .utils import weight_decay

__all__ = ['resnetcif20', 'resnetcif20b','resnetcif32', 'resnetcif44', 'resnetcif56', 'resnetcif110', 'resnetcif1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.compressible_conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.compressible_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.option = option

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(OrderedDict([
                    ('compressible_conv2d', nn.Conv2d(in_planes, self.expansion * planes, kernel_size=2, stride=stride, bias=False)),
                    ('batch_norm', nn.BatchNorm2d(self.expansion * planes))
                ]))

    def forward(self, x):
        # print('input to compressible_conv1:', x.size())
        out = F.relu(self.bn1(self.compressible_conv1(x)))
        # print('after compressible_conv1:', out.size(), self.compressible_conv1)
        out = self.bn2(self.compressible_conv2(out))
        # print('after compressible_conv2:', out.size())
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option='A'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.compressible_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option)
        self.compressible_linear = nn.Linear(64, num_classes)

        self.output = nn.Sequential(
            self.compressible_conv1,
            self.bn1,
            nn.ReLU(True),
            self.layer1,
            self.layer2,
            self.layer3,
            LambdaLayer(lambda out: F.avg_pool2d(out, out.size()[3])),
            LambdaLayer(lambda out: out.view(out.size(0), -1)),
            self.compressible_linear
        )

        self.weight_decay = weight_decay(self)
        self.loss = lambda x, target: nn.CrossEntropyLoss()(x, target) + 1e-4*self.weight_decay()


        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.output(x)

    def compressible_modules(self):
        for name, module in self.named_modules():
            if 'compressible' in name and name not in self.except_:
                yield name, module


def resnetcif20():
    return ResNet(BasicBlock, [3, 3, 3], option='A')

def resnetcif20b():
    return ResNet(BasicBlock, [3, 3, 3], option='B')

def resnetcif32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnetcif44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnetcif56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnetcif110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnetcif1202():
    return ResNet(BasicBlock, [200, 200, 200])