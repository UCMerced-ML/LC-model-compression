#!/usr/bin/env python3
from abc import ABC, abstractmethod
from .utils import diff_based_coding
import numpy as np

class CompressionTypeBase(ABC):
    """"Base type for all compressions (aka C-steps) in the framework."""
    def __init__(self):
        self.__mu = 0
        self._state = None
        self.__info = {}
        self.step_number = 0

    @abstractmethod
    def compress(self, data):
        pass

    @property
    def mu(self):
        return self.__mu

    @property
    def state_dict(self):
        return self._state

    @property
    def info(self):
        return self.__info

    @info.setter
    def info(self, new_info):
        self.__info = new_info

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abstractmethod
    def uncompress_state(self):
        pass

    @abstractmethod
    def count_params(self):
        """
        Counts the number of parameters in the current compressed weights
        """
        pass

    @abstractmethod
    def count_param_bits(self):
        """
        Counts the number of bits required to save the current compressed weights
        """
        pass

    @mu.setter
    def mu(self, mu):
        if mu < 0:
            raise ValueError("mu should be a non-negative floating point number")
        else:
            self.__mu = mu


class PruningTypeBase(CompressionTypeBase):
    """A base type for all pruning in framework. Semantically it does the same, but calling compress() to
    actually prune the values seemed to be incorrect. Therefore we defined a new baseclass that implements compress()
    and prune() methods, and compress() internally calls prune() """
    def __init__(self):
        super(PruningTypeBase, self).__init__()

    def compress(self, data):
        """
        The compression is pruning. Therefore, this method simply calls self.prune(data).
        We have this separation because self.prune() is semantically more suitable for
        nature of this compression techniques.

        :param data: The data that should be pruned (compressed).
        """
        return self.prune(data)

    @abstractmethod
    def prune(self, data):
        pass

    def load_state_dict(self, state_dict):
        self._state = state_dict

    def count_params(self):
        if "remaining_values" in self._state:
            return np.size(self.state_dict["remaining_values"])
        else:
            raise Error("Cannot count the number of compressed params: no compression was performed")

    def count_param_bits(self):
        if "remaining_values" in self._state:
            pruned = self.uncompress_state()
            indx = np.flatnonzero(pruned)
            diff_indx = indx[1:] - indx[:-1]
            storage_bits = diff_based_coding(diff_indx, correction_precision=32,
                                             diff_bases=range(2, 33))
            return storage_bits
        else:
            raise Error("Cannot count the number of compressed params: no compression was performed")


    def uncompress_state(self):
        remaining_indx = self.state_dict['remaining_indx']
        remaining_values = self.state_dict["remaining_values"]
        pruned = np.zeros(self.state_dict['len'])
        pruned[remaining_indx] = remaining_values.astype(np.float32)

        return pruned
