#!/usr/bin/env python3
from abc import ABC, abstractmethod


class ViewBase(ABC):
    """
    This is a base class for all views in the framework.
    """
    @property
    @abstractmethod
    def shape(self):
        """Gives the shape of the packed object"""
        pass

    @property
    @abstractmethod
    def list(self):
        """Returns all tensors constructing the packed objects as a list"""
        pass

    @property
    @abstractmethod
    def original_form(self):
        pass

    def pack(self):
        """Call to this method should return a numpy-array object"""
        pass

    def pack_(self, obj):
        pass

    def unpack(self, obj):
        """Re-instates numpy-array object into the original blocks it came from"""
        pass

    def unpack_(self, obj):
        pass


class Parameter(ABC):
    """
    This is a base class for a triplets of (w, Δ(ϴ), λ) and associated penalty computation for the purposes of
    LC algorithm.
    """
    def __init__(self, w, w_view, delta_theta, target, target_view, lambda_):
        """
        :param w: weights vector
        :param w_view: view of the w in NN, e.g., a tensor
        :param delta_theta: compressed weights vector
        :param target_view: view for NN
        :param lambda_: Lagrange multipliers vector
        """
        self.w = w
        self.delta_theta = delta_theta
        self.lambda_ = lambda_
        self.target = target
        if isinstance(w_view, ViewBase) and \
                isinstance(target_view, ViewBase):
            self.w_view = w_view
            self.target_view = target_view
        else:
            raise TypeError("w_view and delta_theta_view must be of ViewBase type.")

    @abstractmethod
    def retrieve(self):
        """retrieve from the outside world"""
        pass

    @abstractmethod
    def release(self):
        """release to the outside world"""
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def lc_penalty(self):
        pass

    def vector_to_compression_view(self, vector, compression_view):
        original_view = self.w_view.unpack_(vector)
        data_for_compression = compression_view.pack_(original_view)
        return data_for_compression

    def compression_view_to_vector(self, compression_view_data, compression_view):
        delta_theta_ov = compression_view.unpack_(compression_view_data)
        delta_theta = self.w_view.pack_(delta_theta_ov)
        return delta_theta