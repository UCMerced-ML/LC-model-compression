import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np

from tensorpack_additions import MyLMDBSerializer, MyLocallyShuffleData
from tensorpack.dataflow import LMDBSerializer, MultiProcessRunnerZMQ

import math
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

class ExternalInputIterator(object):
  def __init__(self, file_location, batch_size, train=True, shuffle=True, full=False, batch_from_disk=150):
    self.batch_size = batch_size
    self.train = train
    if train:
      self.ds = MyLMDBSerializer.load(file_location, shuffle=shuffle, batch_from_disk=batch_from_disk)
      self.ds = MyLocallyShuffleData(self.ds, buffer_size=10000, shuffle_interval=500)
      self.ds = MultiProcessRunnerZMQ(self.ds, num_proc=1, hwm=10000)
      self.len_ = 1281167
    else:
      self.ds = LMDBSerializer.load(file_location, shuffle=False)
      self.ds = MultiProcessRunnerZMQ(self.ds, num_proc=1, hwm=10000)
      self.len_ = 50000
    self.ds.reset_state()
    self.batches_in_epoch = int(math.ceil(self.len_ / self.batch_size))

  def __iter__(self):
    if not self.train:
      self.iterator = self.ds.__iter__()
    else:
      if not hasattr(self, 'iterator'):
        self.iterator = self.ds.__iter__()
    self.i = 0
    return self

  def __next__(self):
    batch = []
    labels = []

    if self.i >= self.batches_in_epoch:
      raise StopIteration

    for _ in range(self.batch_size):
      jpeg_bytes, label = next(self.iterator)
      batch.append(jpeg_bytes)
      labels.append(np.array([label], dtype = np.uint8))
    self.i += 1
    return batch, labels

  next = __next__


class LightningAug():
  def __init__(self, std, eigval, eigvec, normalize_std):
    self.std = std
    eigval = torch.tensor(eigval, dtype=torch.float32)
    eigvec = torch.tensor(eigvec, dtype=torch.float32) # each column is an eigenvector
    normalize_std=torch.tensor(normalize_std, dtype=torch.float32)
    self.mat = torch.mm(eigvec, torch.diag(eigval/normalize_std)).cuda()

  def __call__(self, batch):
    n_img = batch.size(0)
    alphas = self.std*torch.randn(3, n_img, dtype=batch.dtype).cuda()
    noise = torch.mm(self.mat, alphas).transpose(0,1)
    batch.data.add_(noise[:,:,None,None])
    return batch

class CaffeTrainPipe(Pipeline):
  def __init__(self, batch_size, num_threads, device_id, file_location, resize_size, crop_size, dali_cpu=False,
               full_sized=False, batch_from_disk=150, square=False, shuffle=True):
    super(CaffeTrainPipe, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=2)

    self.eii = ExternalInputIterator(file_location, batch_size, train=True, shuffle=shuffle,
                                     full=full_sized, batch_from_disk=batch_from_disk)
    self.eii_iterator = iter(self.eii)
    self.input = ops.ExternalSource()
    self.input_label = ops.ExternalSource()
    # let user decide which pipeline works him bets for RN version he runs
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
    # without additional reallocations
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    self.decode = self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,)
    self.full_sized = full_sized
    self.square = square
    resize_not_needed = (not square) and (not full_sized)
    self.resize_needed = not resize_not_needed
    if self.resize_needed:
      print(f"Resize will happen for TRAIN dataset, full_sized={full_sized}, square={square}")
      if square:
        self.res = ops.Resize(device=dali_device, resize_x=resize_size, resize_y=resize_size, interp_type=types.INTERP_TRIANGULAR)
      else:
        self.res = ops.Resize(device=dali_device, resize_shorter=resize_size, interp_type=types.INTERP_TRIANGULAR)
    else:
      print(f"No resize will happen for for TRAIN dataset, full_sized={full_sized}, square={square}")
    self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NCHW,
                                        crop=(crop_size, crop_size),
                                        image_type=types.RGB,
                                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    self.coin = ops.CoinFlip(probability=0.5)
    self.uniform = ops.Uniform(range=(0.0, 1.0))
    print('DALI "{0}" variant'.format(dali_device))

  def define_graph(self):
    self.jpegs = self.input()
    self.labels = self.input_label()
    images = self.decode(self.jpegs)
    if self.resize_needed:
      images = self.res(images)
    output = self.cmnp(images.gpu(), mirror=self.coin(), crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
    return [output, self.labels]

  def iter_setup(self):
    try:
      (images, labels) = self.eii_iterator.next()
      self.feed_input(self.jpegs, images)
      self.feed_input(self.labels, labels)
    except StopIteration:
      self.eii_iterator = iter(self.eii)
      raise StopIteration


class CaffeValPipe(Pipeline):
  def __init__(self, batch_size, num_threads, device_id, data_dir, crop_size, resize_size, full_sized=False, square=False):
    self.eii = ExternalInputIterator(data_dir, batch_size, train=False, full=full_sized)
    self.eii_iterator = iter(self.eii)
    super(CaffeValPipe, self).__init__(batch_size, num_threads, device_id)
    self.input = ops.ExternalSource()
    self.input_label = ops.ExternalSource()
    self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
    self.full_sized = full_sized
    resize_not_needed = (not square) and (not full_sized)
    self.resize_needed = not resize_not_needed
    if self.resize_needed:
      print(f"Resize will happen for VAL dataset, full_sized={full_sized}, square={square}")
      if square:
        self.res = ops.Resize(device='gpu', resize_x=resize_size, resize_y=resize_size, interp_type=types.INTERP_TRIANGULAR)
      else:
        self.res = ops.Resize(device='gpu', resize_shorter=resize_size, interp_type=types.INTERP_TRIANGULAR)
    else:
      print(f"No resize will happen for VAL dataset, full_sized={full_sized}, square={square}")
    self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NCHW,
                                        crop=(crop_size, crop_size),
                                        image_type=types.RGB,
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255])

  def define_graph(self):
    self.jpegs = self.input()
    self.labels = self.input_label()
    images = self.decode(self.jpegs)
    if self.resize_needed:
      images = self.res(images)
    output = self.cmnp(images)
    return [output, self.labels]

  def iter_setup(self):
    try:
      (images, labels) = self.eii_iterator.next()
      self.feed_input(self.jpegs, images)
      self.feed_input(self.labels, labels)
    except StopIteration:
      self.eii_iterator = iter(self.eii)
      raise StopIteration