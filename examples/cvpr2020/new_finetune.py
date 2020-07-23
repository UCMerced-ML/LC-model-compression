import torch
from torch import nn
from collections import OrderedDict
from scipy.linalg import svd
import numpy as np

class FullRankException(Exception):
    pass

class RankNotEfficientException(Exception):
    pass


def linear_layer_reparametrizer(sub_module, conv_scheme='scheme_1'):
    W = sub_module.weight.data.cpu().numpy()

    init_shape = None
    n,m,d1,d2 = None, None, None, None
    if isinstance(sub_module, nn.Conv2d):
        if conv_scheme == 'scheme_1':
            init_shape = W.shape
            reshaped = W.reshape([init_shape[0], -1])
            W = reshaped
        elif conv_scheme == 'scheme_2':
            [n, m, d1, d2] = W.shape
            swapped = W.swapaxes(1, 2)
            reshaped = swapped.reshape([n * d1, m * d2])
            W = reshaped

    u, s, v = svd(W, full_matrices=False)
    from numpy.linalg import matrix_rank

    r = sub_module.rank_ if hasattr(sub_module, 'rank_') \
        else sub_module.selected_rank_ if hasattr(sub_module, 'selected_rank_') \
        else int(matrix_rank(W))
    print(r)
    if r < np.min(W.shape):
        diag = np.diag(s[:r] ** 0.5)
        U = u[:, :r] @ diag
        V = diag @ v[:r, :]
        print(U.shape, V.shape)
        new_W = U @ V


        from numpy.linalg import norm
        print('Fro. norm of W: ', norm(W))
        print('Fro. norm of diff: ', norm(W - new_W))


        m,n = W.shape
        if r > np.floor(m*n/(m+n)):
            raise RankNotEfficientException("Selected rank doesn't contribute to any savings")
        bias = sub_module.bias is not None
        if isinstance(sub_module, nn.Linear):
            l1 = nn.Linear(in_features=sub_module.in_features, out_features=r, bias=False)
            l2 = nn.Linear(in_features=r, out_features=sub_module.out_features, bias=bias)
            l1.weight.data = torch.from_numpy(V)
            l2.weight.data = torch.from_numpy(U)
            if bias:
                l2.bias.data = sub_module.bias.data
            return l1, l2
        else:
            if conv_scheme == 'scheme_1':
                l1 = nn.Conv2d(in_channels=sub_module.in_channels,
                               out_channels=r,
                               kernel_size=sub_module.kernel_size,
                               stride=sub_module.stride,
                               padding=sub_module.padding,
                               dilation=sub_module.dilation,
                               groups=sub_module.groups,
                               bias=False
                               )
                l2 = nn.Conv2d(in_channels=r, out_channels=sub_module.out_channels,
                               kernel_size=1,
                               bias=bias)
                l1.weight.data = torch.from_numpy(V.reshape([-1, *init_shape[1:]]))
                l2.weight.data = torch.from_numpy(U[:, :, None, None])

                if bias:
                    l2.bias.data = sub_module.bias.data

                return l1, l2
            elif conv_scheme == 'scheme_2':
                raise NotImplementedError("We did not implement scheme-2 for CIFAR10.")


def reparametrization_helper(list_of_modules, conv_scheme, old_weight_decay=True):
    new_sequence = []
    items = list_of_modules.items()
    decayed_values_repar = []
    decayed_valued_old = []
    from lc.models.torch.resnetcif import BasicBlock
    for i, (name, sub_module) in enumerate(items):
        print(name, sub_module)

        if isinstance(sub_module, nn.Sequential):
            dv_repar_sub, dv_old_sub, nseq_sub = reparametrization_helper(sub_module._modules,
                                                                          conv_scheme=conv_scheme,
                                                                          old_weight_decay=old_weight_decay)
            new_sequence.append((name, nn.Sequential(OrderedDict(nseq_sub))))
            decayed_values_repar.extend(dv_repar_sub)
            decayed_valued_old.extend(dv_old_sub)

        elif isinstance(sub_module, nn.Linear) or isinstance(sub_module, nn.Conv2d):
            try:
                l1, l2 = linear_layer_reparametrizer(sub_module, conv_scheme=conv_scheme)
                new_sequence.append((name + '_V', l1))
                new_sequence.append((name + '_U', l2))
                decayed_values_repar.append((l1, l2))

            except Exception as e:
                print(str(e))
                new_sequence.append((name, sub_module))
                decayed_valued_old.append(sub_module.weight)
        elif isinstance(sub_module, BasicBlock):
            try:
                l1, l2 = linear_layer_reparametrizer(sub_module.compressible_conv1,
                                                     conv_scheme=conv_scheme)
                decayed_values_repar.append((l1, l2))
                sub_module.compressible_conv1 = nn.Sequential(
                    OrderedDict([('compressible_conv1'+'_V', l1), ('compressible_conv1'+'_u', l2)]))
            except Exception as e:
                print(str(e),  name, 'compressible_conv1')
                decayed_valued_old.append(sub_module.compressible_conv1.weight)

            try:
                l1, l2 = linear_layer_reparametrizer(sub_module.compressible_conv2, conv_scheme=conv_scheme)
                decayed_values_repar.append((l1, l2))
                sub_module.compressible_conv2 = nn.Sequential(
                    OrderedDict([('compressible_conv2'+'_V', l1), ('compressible_conv2'+'_u', l2)]))
            except Exception as e:
                print(str(e), name, 'compressible_conv2')
                decayed_valued_old.append(sub_module.compressible_conv2.weight)

            if len(sub_module.shortcut._modules) > 0:
                dv_repar_sub, dv_old_sub, nseq_sub \
                    = reparametrization_helper(sub_module._modules,
                                               conv_scheme=conv_scheme,
                                               old_weight_decay=old_weight_decay)
                sub_module.shortcut = nn.Sequential(OrderedDict(nseq_sub))
                decayed_values_repar.extend(dv_repar_sub)
                decayed_valued_old.extend(dv_old_sub)

            new_sequence.append((name, sub_module))
            if old_weight_decay:
                decayed_valued_old.append(sub_module.bn1.weight)
                decayed_valued_old.append(sub_module.bn2.weight)

        else:
            new_sequence.append((name, sub_module))
            if old_weight_decay and hasattr(sub_module, 'weight'):
                decayed_valued_old.append(sub_module.weight)
    return decayed_values_repar, decayed_valued_old, new_sequence


def reparametrize_low_rank(model, old_weight_decay=True):
    decayed_values_repar, decayed_valued_old, new_sequence \
        = reparametrization_helper(model.output._modules, conv_scheme='scheme_1', old_weight_decay=old_weight_decay)
    model.output = nn.Sequential(OrderedDict(new_sequence))

    for xi in decayed_valued_old:
        print(xi.shape)
    for xi,xj in decayed_values_repar:
        print(xi.weight.shape, xj.weight.shape)

    def weight_decay():
        sum_ = torch.autograd.Variable(torch.FloatTensor([0.0]).cuda())
        for x in decayed_valued_old:
            sum_ += torch.sum(x**2)
        for v,u in decayed_values_repar:
            v = v.weight
            u = u.weight
            u_ = u.view(u.size()[0], -1)
            v_ = v.view(u_.size()[1], -1)
            sum_ += torch.sum(torch.matmul(u_,v_)**2)
        return sum_
    model.weight_decay = weight_decay
    print(new_sequence)