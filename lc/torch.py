import torch
import torch.nn as nn
import numpy as np
from lc import Parameter
from lc.base_types import ViewBase
import torch
import numpy as np


class AsVector(ViewBase):
    def __init__(self, list_or_item):
        """
        Packs single tensor of any shape or list of tensors of any shape into one vector.
        :param list_or_item:
        """
        if isinstance(list_or_item, list):
            if all(map(lambda x_getter: isinstance(x_getter(), torch.Tensor), list_or_item)):
                self.tensors = list_or_item
            else:
                raise AttributeError("Not all items in the list are Torch.Tensor-s")
        elif isinstance(list_or_item(), torch.Tensor):
            item = list_or_item
            self.tensors = [item]
        else:
            raise AttributeError("Supplied argument is not a list of torch.Tensor-s nor a single torch.Tensor")

        self.list_or_item = list_or_item
        self.shapes = []
        self.indexes = []
        self.total_len = 0
        start = 0
        for tensor_getter in self.tensors:
            tensor = tensor_getter()
            self.shapes.append(tensor.detach().cpu().numpy().shape)
            flat_len = np.prod(self.shapes[-1])
            end = start + flat_len
            self.indexes.append((start, end))
            self.total_len += flat_len
            start = end

    @property
    def list(self):
        return self.tensors[:]

    @property
    def original_form(self):
        return self.list_or_item

    @property
    def shape(self):
        return np.zeros(self.total_len).shape

    def pack(self):
        vector = np.zeros(self.total_len)
        for tensor_getter, (start, end) in zip(self.tensors, self.indexes):
            tensor = tensor_getter()
            vector[start:end] = tensor.detach().cpu().numpy().flatten()
        return vector

    def pack_(self, list_or_item):
        list_ = list_or_item
        if not isinstance(list_or_item, list):
            list_ = [list_or_item]
        vector = np.zeros(self.total_len)
        for tensor, (start, end) in zip(list_, self.indexes):
            vector[start:end] = tensor.detach().cpu().numpy().flatten()
        return vector

    def unpack(self, vector):
        for tensor_getter, (start, end), shape in zip(self.tensors, self.indexes, self.shapes):
            tensor=tensor_getter()
            unpacked_tensor = torch.from_numpy(vector[start:end].reshape(shape)).to(tensor.device, dtype=torch.float32)
            tensor.data = unpacked_tensor

    def unpack_(self, vector):
        list_ = []
        for tensor_getter, (start, end), shape in zip(self.tensors, self.indexes, self.shapes):
            tensor = tensor_getter()
            unpacked_tensor = torch.from_numpy(vector[start:end].reshape(shape)).to(tensor.device, dtype=torch.float32)
            list_.append(unpacked_tensor)
        return list_


class AsIs(ViewBase):
    def __init__(self, list_or_item):
        """
        Packs a single tensor of any shape or a list of tensors of any shape as is, i.e., with no reshaping nor merging
        :param list_or_item:
        """
        self.single = False
        if isinstance(list_or_item, list):
            if all(map(lambda x: isinstance(x(), torch.Tensor), list_or_item)):
                self.tensors = list_or_item
            else:
                raise AttributeError("Not all items in the list are Torch.Tensor-s")
        elif isinstance(list_or_item(), torch.Tensor):
            self.single = True
            item = list_or_item
            self.tensors = [item]
        else:
            raise AttributeError("Supplied argument is not a list of torch.Tensor-s nor a single torch.Tensor")

        self.list_or_item = list_or_item
        self.shapes = []
        self.indexes = []
        self.total_len = 0
        start = 0
        for tensor_getter in self.tensors:
            tensor=tensor_getter()
            self.shapes.append(tensor.detach().cpu().numpy().shape)
            flat_len = np.prod(self.shapes[-1])
            end = start + flat_len
            self.indexes.append((start, end))
            self.total_len += flat_len
            start = end

    @property
    def list(self):
        return self.tensors[:]

    @property
    def original_form(self):
        return self.list_or_item

    @property
    def shape(self):
        return np.zeros(self.total_len).shape

    def pack(self):
        l_ = []
        for tensor_getter in self.tensors:
            tensor = tensor_getter()
            l_.append(tensor.detach().cpu().numpy())
        if self.single:
            return l_[0]
        return l_

    def pack_(self, list_):
        l_ = []
        for tensor in list_:
            l_.append(tensor.detach().cpu().numpy())
        if self.single:
            return l_[0]
        return l_

    def unpack(self, l_):
        if not isinstance(l_, list):
            l_ = [l_]
        for tensor_getter, copy in zip(self.tensors, l_):
            tensor = tensor_getter()
            tensor.data = torch.from_numpy(copy).to(tensor.device, dtype=torch.float32)

    def unpack_(self, l_):
        if not isinstance(l_, list):
            l_ = [l_]
        new_l = []
        for tensor_getter, copy in zip(self.tensors, l_):
            tensor = tensor_getter()
            new_l.append(torch.from_numpy(copy).to(tensor.device, dtype=torch.float32))
        return new_l


class ParameterTorch(Parameter):
    def __init__(self, single_or_list_of_tensors, device, bn_after=False):
        w_view = AsVector(single_or_list_of_tensors)
        w = np.zeros(w_view.shape)

        d_theta_list = []
        for tensor_get in w_view.list:
            d_theta = nn.Parameter(data=torch.zeros_like(tensor_get()), requires_grad=False)
            d_theta_list.append(lambda x=d_theta: x)
        target_view = AsVector(d_theta_list)

        super(ParameterTorch, self)\
            .__init__(w=w,
                      w_view=w_view,
                      delta_theta=np.zeros_like(w),
                      target=np.zeros_like(w),
                      target_view=target_view,
                      lambda_=np.zeros_like(w))

        self.device = device
        self.is_eval = False
        self.d_theta_list = d_theta_list
        self.bn_after = bn_after

    def retrieve(self, full=False):
        self.w = self.w_view.pack()
        # Although, we can run the following line, we know that delta_theta is changed only by LC
        if full:
            self.delta_theta = self.target_view.pack()
        print("from retrieve", np.linalg.norm(self.w)**2)

    def release(self):
        # Although, we don't really change the w_view, sometimes we do, e.g,. when we do rescaling of weights.
        # In such cases we need to propagate changes back to GPU backend. Since we do not track modification to self.w
        # we are pushing back everything to be extra safe.
        self.w_view.unpack(self.w)
        self.target_view.unpack(self.target)

    def eval(self):
        if self.is_eval:
            return
        else:
            self.w_view.unpack(self.delta_theta)
            self.is_eval = True

    def train(self):
        if self.is_eval:
            self.w_view.unpack(self.w)
            self.is_eval = False
        else:
            return

    def lc_penalty(self):
        loss_ = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for w_getter, target_getter in zip(self.w_view.list, self.target_view.list):
            loss_ += torch.sum((w_getter() - target_getter()) ** 2)

        return loss_
