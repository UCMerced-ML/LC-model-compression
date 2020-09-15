import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.model_zoo
import numpy as np
from dataloaders import CaffeValPipe
from collections import OrderedDict
import argparse

class CaffeBNAlexNet(nn.Module):
  def __init__(self, num_classes=1000):
    super(CaffeBNAlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(96, 256, kernel_size=5, padding=2),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(256, 384, kernel_size=3, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 384, kernel_size=3, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Linear(256 * 6 * 6, 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, num_classes),
    )
    self.loss = lambda x, target: nn.CrossEntropyLoss()(x, target)


  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def my_forward(self, x):
    out=x
    for i in range(len(self.features)):
      print(self.features[i])
      out=self.features[i](out)
      print(out.size())


def generate_low_rank_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                             bias=True, padding_mode='zeros', rank=None, scheme='scheme_1'):
  if rank is None:
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)
  if scheme == 'scheme_1':
    l1 = nn.Conv2d(in_channels=in_channels,
                   out_channels=rank,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   groups=groups,
                   bias=False
                   )
    l2 = nn.Conv2d(in_channels=rank, out_channels=out_channels,
                   kernel_size=1,
                   bias=bias)
    return nn.Sequential(OrderedDict([('V', l1), ('U', l2)]))
  elif scheme == 'scheme_2':
    if isinstance(kernel_size, int):
      kernel_size = [kernel_size, kernel_size]

    if isinstance(padding, int):
      padding = [padding, padding]
    if isinstance(stride, int):
      stride = [stride, stride]
    l1 = nn.Conv2d(in_channels=in_channels,
                   out_channels=rank,
                   kernel_size=(1, kernel_size[1]),
                   stride=(1, stride[1]),
                   padding=(0, padding[1]),
                   dilation=dilation,
                   groups=groups,
                   bias=False
                   )
    l2 = nn.Conv2d(in_channels=rank,
                   out_channels=out_channels,
                   kernel_size=(kernel_size[0], 1),
                   padding=(padding[0], 0),
                   stride=(stride[0], 1),
                   bias=bias)

    return nn.Sequential(OrderedDict([('V', l1), ('U', l2)]))


def generate_low_rank_linear(in_features, out_features, bias=True, rank=None):
  if rank is None:
    return nn.Linear(in_features, out_features, bias=bias)
  l1 = nn.Linear(in_features=in_features, out_features=rank, bias=False)
  l2 = nn.Linear(in_features=rank, out_features=out_features, bias=bias)
  return nn.Sequential(OrderedDict([('V', l1), ('U', l2)]))

class CaffeBNLowRankAlexNet(nn.Module):
  def __init__(self, ranks, scheme, num_classes=1000,):
    super(CaffeBNLowRankAlexNet, self).__init__()
    self.features = nn.Sequential(
      generate_low_rank_conv2d(3, 96, kernel_size=11, stride=4, padding=2, rank=ranks[0], scheme=scheme),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      generate_low_rank_conv2d(96, 256, kernel_size=5, padding=2, rank=ranks[1], scheme=scheme),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      generate_low_rank_conv2d(256, 384, kernel_size=3, padding=1, rank=ranks[2], scheme=scheme),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      generate_low_rank_conv2d(384, 384, kernel_size=3, padding=1, rank=ranks[3], scheme=scheme),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      generate_low_rank_conv2d(384, 256, kernel_size=3, padding=1, rank=ranks[4], scheme=scheme),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      generate_low_rank_linear(256 * 6 * 6, 4096, rank=ranks[5]),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      generate_low_rank_linear(4096, 4096, rank=ranks[6]),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      generate_low_rank_linear(4096, num_classes, rank=ranks[7]),
    )
    self.loss = lambda x, target: nn.CrossEntropyLoss()(x, target)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

def validate(model, lmdb_file_location, full_sized=False):
  import time, os
  from nvidia.dali.plugin.pytorch import DALIClassificationIterator

  def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
      maxk = max(topk)
      batch_size = target.size(0)

      _, pred = output.topk(maxk, 1, True, True)
      pred = pred.t()
      correct = pred.eq(target.view(1, -1).expand_as(pred))

      res = []
      for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
      return res

  class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
      self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
      self.meters = meters
      self.prefix = prefix

    def display(self, batch):
      entries = [self.prefix + self.batch_fmtstr.format(batch)]
      entries += [str(meter) for meter in self.meters]
      print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
      num_digits = len(str(num_batches // 1))
      fmt = '{:' + str(num_digits) + 'd}'
      return '[' + fmt + '/' + fmt.format(num_batches) + ']'

  class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
      self.name = name
      self.fmt = fmt
      self.reset()

    def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

    def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

    def __str__(self):
      fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
      return fmtstr.format(**self.__dict__)

  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
    50000 // 500,
    [batch_time, losses, top1, top5],
    prefix='Test: ')

  def test_loader():
    pipe = CaffeValPipe(batch_size=500, num_threads=4, device_id=0,
                        data_dir=os.path.expanduser(lmdb_file_location), crop_size=227, resize_size=256,
                        full_sized=full_sized)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.eii.len_), auto_reset=True, fill_last_batch=False)
    for data in val_loader:
      images = data[0]["data"]
      target = data[0]["label"].squeeze().cuda().long()
      yield images, target

  model.cuda()
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(test_loader()):
      target = target.cuda(non_blocking=True)
      images = images.cuda(non_blocking=True)
      # compute output
      output = model(images)
      loss = model.loss(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 10 == 0:
        progress.display(i)

    print(f' * Top-1 err: {100-top1.avg:.3f}%  Top-5 err: {100-top5.avg:.3f}%')


sha256_and_ranks = {
    "scheme_2_v4": ("d89ee385", "https://ucmerced.box.com/shared/static/81r59yj437ngyr4t4m9gfpx08y3k5rej.th",
                    [6,  26,  70,  74,  96,  534,  423, 256]),
    "scheme_2_v3": ("56982c03", "https://ucmerced.box.com/shared/static/stz6t6jd7hioktyw0kngypbpnkh6hrda.th",
                    [9,  29,  85,  92, 118,  762,  724, 256]),
    "scheme_2_v2": ("a126ab06", "https://ucmerced.box.com/shared/static/mlso238v2pa9taxqkf0l2rsyjhyi56jn.th",
                    [10, 35, 107, 117, 150, 1050, 1107, 256]),
    "scheme_2_v1": ("bf688dc1", "https://ucmerced.box.com/shared/static/nqsgh4bbcvih8gjcfsqhyb2tw1ojwpho.th",
                    [13, 45, 146, 172, 200, 1726, 1841, 256]),

    "scheme_1_v1": ("f15f3f59", "https://ucmerced.box.com/shared/static/kf1c2yiq4fdpn1qhx783vzva3o4h09xh.th",
                    [33, 52, 134, 135, 159, 1703, 1845, 256]),
    "scheme_1_v2": ("8a201fc0", "https://ucmerced.box.com/shared/static/xkyu8j6iq4gerh8dsjo7jeqi5yj9iwla.th",
                    [25, 36, 84,   71, 89,   751,  724, 256]),
    "scheme_1_v3": ("97c0439b", "https://ucmerced.box.com/shared/static/xwi9i1laqvxogd45kxlucycm46xdk8u4.th",
                    [23, 34, 75,   64, 82,   598,  498, 256]),
  }

def low_rank_alexnet(scheme, tag, pretrained=False):
  sha256_first8, link, ranks = sha256_and_ranks[f"{scheme}_{tag}"]
  model = CaffeBNLowRankAlexNet(scheme=scheme, ranks=ranks)
  if pretrained:
    model_state = torch.utils.model_zoo.load_url(link, progress=True)
    model.load_state_dict(model_state)
  return model


def load_and_validate(scheme, tag, lmdb_file_loation):
  model = low_rank_alexnet(scheme, tag, pretrained=True)
  validate(model, lmdb_file_loation)


if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Low-rank AlexNet validation script')
  parser.add_argument('--lmdb_file', type=str)
  args = parser.parse_args()

  load_and_validate("scheme_2", "v1", args.lmdb_file)
  load_and_validate("scheme_2", "v2", args.lmdb_file)
  load_and_validate("scheme_2", "v3", args.lmdb_file)
  load_and_validate("scheme_2", "v4", args.lmdb_file)

  load_and_validate("scheme_1", "v1", args.lmdb_file)
  load_and_validate("scheme_1", "v2", args.lmdb_file)
  load_and_validate("scheme_1", "v3", args.lmdb_file)
