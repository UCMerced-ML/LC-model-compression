import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
import os
from torch.utils.data import TensorDataset, DataLoader


def mnist_data():
    if os.path.isfile('./datasets/mnist_dataset.pkl'):
        return torch.load('./datasets/mnist_dataset.pkl')
    else:
        mnist_train = torchvision.datasets.MNIST(root='./datasets/', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(root='./datasets/', train=False, download=True)

        train_data = mnist_train.data.to(torch.float) / 255.
        test_data = mnist_test.data.to(torch.float) / 255.
        mean_image = torch.mean(train_data, dim=0)

        train_data -= mean_image
        test_data -= mean_image

        train_labels = mnist_train.targets
        test_labels = mnist_test.targets

        our_mnist = {
            'train_data': train_data, 'test_data': test_data,
            'train_labels': train_labels, 'test_labels': test_labels
        }
        torch.save(our_mnist, './datasets/mnist_dataset.pkl')
        return our_mnist


def data_loader(batch_size=256, dataset='',n_workers=0):
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=False)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=False)

        return train_loader, val_loader
    elif dataset=='CIFAR10_no_mod':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=False)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=False)

        return train_loader, val_loader
    elif dataset =='MNIST':
        data = mnist_data()
        print(data['train_data'].size())
        train_data = TensorDataset(data['train_data'], data['train_labels'])
        test_data = TensorDataset(data['test_data'], data['test_labels'])
        print(data['test_data'].size())
        train_loader = DataLoader(train_data, num_workers=n_workers, batch_size=batch_size, shuffle=True,
                                  pin_memory=False)
        test_loader = DataLoader(test_data, num_workers=n_workers, batch_size=batch_size, shuffle=False,
                                 pin_memory=False)

        return train_loader, test_loader


def format_time(seconds):
    if seconds < 60:
        return '{:.1f}s.'.format(seconds)
    if seconds < 3600:
        return '{:d}m. {}'.format(int(seconds//60), format_time(seconds%60))
    if seconds < 3600*24:
        return '{:d}h. {}'.format(int(seconds//3600), format_time(seconds%3600))

    return '{:d}d. {}'.format(int(seconds//(3600*24)), format_time(seconds%(3600*24)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder():
    def __init__(self):
        pass

    def record(self, tag, value):
        if hasattr(self, tag):
            self.__dict__[tag].append(value)
        else:
            self.__dict__[tag] = [value]


def compute_acc_loss(forward_func, data_loader):
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(data_loader):
        with torch.no_grad():
            target = target.cuda()
            score, loss = forward_func(x.cuda(), target)
            _, pred_label = torch.max(score.data, 1)
            correct_cnt += (pred_label == target.data).sum().item()
            ave_loss += loss.data.item() * len(x)
    accuracy = correct_cnt * 1.0 / len(data_loader.dataset)
    ave_loss /= len(data_loader.dataset)
    return accuracy, ave_loss

def add_flops_counting_methods(net_main_module):
    """Adds flops counting functions to an existing model. After that
    the flops count should be activated and the model should be run on an input
    image.

    Example:

    fcn = add_flops_counting_methods(fcn)
    fcn = fcn.cuda().train()
    fcn.start_flops_count()


    _ = fcn(batch)

    fcn.compute_average_flops_cost() / 1e9 / 2 # Result in GFLOPs per image in batch

    Important: dividing by 2 only works for resnet models -- see below for the details
    of flops computation.

    Attention: we are counting multiply-add as two flops in this work, because in
    most resnet models convolutions are bias-free (BN layers act as bias there)
    and it makes sense to count muliply and add as separate flops therefore.
    This is why in the above example we divide by 2 in order to be consistent with
    most modern benchmarks. For example in "Spatially Adaptive Computatin Time for Residual
    Networks" by Figurnov et al multiply-add was counted as two flops.

    This module computes the average flops which is necessary for dynamic networks which
    have different number of executed layers. For static networks it is enough to run the network
    once and get statistics (above example).

    Implementation:
    The module works by adding batch_count to the main module which tracks the sum
    of all batch sizes that were run through the network.

    Also each convolutional layer of the network tracks the overall number of flops
    performed.

    The parameters are updated with the help of registered hook-functions which
    are being called each time the respective layer is executed.

    Parameters
    ----------
    net_main_module : torch.nn.Module
        Main module containing network

    Returns
    -------
    net_main_module : torch.nn.Module
        Updated main module with new methods/attributes that are used
        to compute flops.
    """

    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding varialbles necessary for masked flops computation
    #net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module

def compute_average_flops_cost(self,quant_k=False):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__

    flops_sum = 0
    fp_mults = 0
    fp_adds = 0
    fp_quant_multp = 0
    for module in self.modules():

        if hasattr(module, '__flops_handle__'):
            flops_sum += module.__flops__
            fp_mults += module.__fpmults__
            fp_adds += module.__fpadds__

        if quant_k and hasattr(module, '__flops_handle__'):
            fp_quant_multp +=module.__fpmults_quant__


    print(flops_sum, fp_mults, fp_adds, fp_quant_multp)
    return flops_sum / batches_count, fp_mults/batches_count, fp_adds/batches_count, fp_quant_multp/batches_count

def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)

def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)

def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)

def conv_flops_counter_hook(conv_module, input, output):

    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels

    # We count multiply-add as 2 flops
    conv_per_position_flops = 2 * kernel_height * kernel_width * in_channels * out_channels

    active_elements_count = batch_size * output_height * output_width

    if hasattr(conv_module, '__quantk__'):
        per_postion_mult = out_channels*conv_module.__quantk__
        conv_module.__fpmults_quant__ = per_postion_mult*active_elements_count

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:
        # do not count biases, as they will be merged to BN (usually)
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops #+ conv_per_position_flops*bias_flops

    conv_module.__flops__ += overall_flops
    conv_module.__fpmults__ += overall_conv_flops*0.5
    conv_module.__fpadds__ += overall_conv_flops*0.5#+conv_per_position_flops*bias_flops
    conv_module.multiplier = output_height * output_width

def linear_flops_counter_hook(linear_module, input, output):

    # Can have multiple inputs, getting the first one
    input = input[0]
    batch_size = input.shape[0]
    in_features = linear_module.in_features
    out_features = linear_module.out_features

    bias_flops = 0
    if linear_module.bias is not None:
        bias_flops = out_features

    if hasattr(linear_module, '__quantk__'):
        linear_module.__fpmults_quant__ = out_features * linear_module.__quantk__*batch_size

    linear_module.__flops__ += 2 * batch_size * (in_features * out_features + bias_flops)
    linear_module.__fpmults__ += batch_size * in_features * out_features
    linear_module.__fpadds__ += batch_size*(in_features*out_features+bias_flops)
    linear_module.multiplier = 1

def batch_counter_hook(module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    module.__batch_counter__ += batch_size

def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0

def add_batch_counter_hook_function(module):

    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle

def remove_batch_counter_hook_function(module):

    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()

        del module.__batch_counter_handle__

def add_flops_counter_variable_or_reset(module):

    if isinstance(module, torch.nn.Conv2d) \
            or isinstance(module, torch.nn.Linear):
        module.__flops__ = 0
        module.__fpmults__ = 0
        module.__fpadds__ = 0
        module.__fpmults_quant__= 0

def add_flops_counter_hook_function(module):
    if isinstance(module, torch.nn.Conv2d):
        if hasattr(module, '__flops_handle__'):
            return

        handle = module.register_forward_hook(conv_flops_counter_hook)
        module.__flops_handle__ = handle

    elif isinstance(module, torch.nn.Linear):
        if hasattr(module, '__flops_handle__'):
            return

        handle = module.register_forward_hook(linear_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if isinstance(module, torch.nn.Conv2d):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__

    elif isinstance(module, torch.nn.Linear):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__