#!/usr/bin/env python3
from .base_types import PruningTypeBase
import numpy as np

"""
References:
[1] Miguel Á. Carreira-Perpiñán and Yerlan Idelbayev
    “Learning-Compression” Algorithms for Neural Net Pruning, CVPR2018,
    https://doi.org/10.1109/CVPR.2018.00890
"""


class ConstraintL0Pruning(PruningTypeBase):

    def __init__(self, kappa, precision='float'):
        """
        This function constructs an object that returns an exact solution of the
            min_θ ‖w-θ‖² s.t. ‖θ‖₀ ≤ κ
        where
            ‖.‖  is an l2 norm
            ‖.‖₀ is an l0 norm, i.e. number of non zero items in a vector.
            κ    is a user-defined constant

        More details in [1]

        :param kappa: is the κ in the equation above
        """
        super(ConstraintL0Pruning, self).__init__()
        self.kappa = kappa
        self.precision = (32 if precision == 'float' else precision)

        if self.precision < 16:
            codebook = 2 ** self.precision
            from .quantization import AdaptiveQuantization
            self.quantizer = AdaptiveQuantization(k=codebook)

    def prune(self, data):
        """
        Implementation details:
        We need to select top-κ values of the array, i.e., need a selection algorithm.
        The selection algorithm should be stable in order to preserve the order among non-pruned values
        between pruning-steps. Unfortunately, there is no stable selection algorithm implemented in numpy,
        therefore we use a stable sort and then select top-kappa items. The only stable sort available
        in numpy is mergesort, therefore complexity of this function is O(N log N) where N is the size of the
        input data.

        :param data: is the w in the equation above (see __init__)
        """
        print("L0 constrained pruning.")
        pruned = np.zeros_like(data)
        indx = np.argsort(np.abs(data), kind='mergesort')
        remaining_indx = indx[-self.kappa:]
        remaining_values = data[remaining_indx]
        if self.precision == 'float' or self.precision == 32:
            # no change as everything is already float32
            pass
        elif self.precision < 16:
            print(f"Quantizing corrections with {2 ** self.precision} codebook values")
            self.quantizer.step_number = self.step_number
            remaining_values = self.quantizer.compress(remaining_values)
        elif self.precision == 16:
            print('Since chosen precision is 16 bits, we will represent it as IEEE 16 bit floating point number')
            remaining_values = remaining_values.astype(np.float16)

        if self.step_number not in self.info:
            self.info[self.step_number] = []

        if self.precision < 16:
            self.info[self.step_number].append({'ratio': np.sum(pruned == 0),
                                                'quantizer_info': self.quantizer.info[self. step_number][-1]})
        else:
            self.info[self.step_number].append({'ratio': np.sum(pruned == 0), 'quantizer_info': None })

        self._state = {
            "remaining_indx": remaining_indx,
            "remaining_values": remaining_values,
            "len": len(data),
            "quantizer_state": self.quantizer.state_dict if self.precision < 16 else None
        }
        pruned[remaining_indx] = remaining_values.astype(np.float32)
        print(f"l0-cons pruning finished. #zeros: {np.mean(pruned == 0)*100:.2f}%")
        return pruned

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.precision < 16:
            self.quantizer.load_state_dict(state_dict['quantizer_state'])


class ConstraintL1Pruning(PruningTypeBase):

    def __init__(self, kappa):
        """
        This function constructs an object that returns an exact solution of the
            min_θ ‖w-θ‖² s.t. ‖θ‖₁ ≤ κ
        where
            ‖.‖  is an l2 norm
            ‖.‖₁ is an l1 norm, i.e. sum of absolute values of a vector;
            κ    is a user-defined constant

        More details in [1] and references therein.

        :param kappa: is the κ in the equation above
        """
        super(ConstraintL1Pruning, self).__init__()
        self.kappa = kappa

    def prune(self, data):
        """
        Complexity of this function is O(N log N)

        :param data: is the w in the equation above (see __init__)
        """
        def subproblem_sol(v, z):
            mu = np.sort(v, kind='mergesort')[::-1]
            cumsum = 0
            ro = -1
            theta = 0
            for j in range(len(mu)):
                cumsum += mu[j]
                if mu[j] < 1 / (j + 1) * (cumsum - z):
                    ro = j
                    theta = 1 / (j + 1) * (cumsum - z)
                    break
            print('identified ro={}, theta={}'.format(ro, theta))
            w = np.maximum(v - theta, 0)
            return w

        u = np.abs(data)
        b = subproblem_sol(u, self.kappa)
        new_x = b * np.sign(data)
        remaining_indx = new_x != 0
        pruned = np.zeros_like(data)
        pruned[remaining_indx] = new_x[remaining_indx]
        remaining_values = new_x[remaining_indx]

        print(f"Number of non-zeros: {len(remaining_values)}")
        print(f"L1-norm:", np.sum(np.abs(pruned)))
        self._state = {
            "remaining_indx": np.where(remaining_indx),
            "remaining_values": remaining_values,
            "len": len(data),
        }

        return pruned


class PenaltyL0Pruning(PruningTypeBase):
    def __init__(self, alpha):
        """
        This function constructs an object that returns and exact solution of the
            min_θ ‖w-θ‖² + 2α/μ*‖θ‖₀
        where
            ‖.‖  is an l2 norm
            ‖.‖₀ is an l0 norm, i.e. number of non zero items in a vector.
            α    is a user-defined constant
            μ    is the parameter of the homotopy function coming from the LC algorithm.

        More details in [1].

        :param alpha: is the α in the equation above
        """
        super(PenaltyL0Pruning, self).__init__()
        self.alpha = alpha

    def prune(self, data):
        if self.mu > 0:
            alph = np.sqrt(2 * self.alpha / self.mu)
            remaining_indx = np.abs(data) > alph
            pruned = np.zeros_like(data)
            pruned[remaining_indx] = data[remaining_indx]
            remaining_values = data[remaining_indx]
            number_of_non_zeros = len(remaining_values)

            print(f"Number of non-zeros: {number_of_non_zeros}")


            self._state = {
                "remaining_indx": np.where(remaining_indx),
                "remaining_values": remaining_values,
                "len": len(data),
            }

            return pruned
        else:
            self._state = {
                "remaining_indx": np.array([]),
                "remaining_values": np.array([]),
                "len": len(data),
            }
            return np.zeros_like(data)


class PenaltyL1Pruning(PruningTypeBase):
    def __init__(self, alpha):
        """
        This function constructs an object that returns solution of the
            min_θ ‖w-θ‖² + 2α/μ*‖θ‖₁
        where
            ‖.‖  is an l2 norm
            ‖.‖₁ is an l1 norm, i.e. sum of absolute values of a vector;
            α    is a user-defined constant
            μ    is the parameter of the homotopy function coming from the LC algorithm.

        More details in [1].

        :param alpha: is α in the equation above
        """
        super(PenaltyL1Pruning, self).__init__()
        self.alpha = alpha

    def prune(self, data):
        """
        Complexity of this function is O(N)

        :param data: is w in the equation above (see __init__)
        """
        if self.mu > 0:
            alph = self.alpha / self.mu
            remaining_indx = np.abs(data) > alph

            pruned = np.zeros_like(data)
            pruned[remaining_indx] = data[remaining_indx]

            # shrinkage happens here
            pruned[pruned > 0] -= alph
            pruned[pruned < 0] += alph

            number_of_non_zeros = np.sum(remaining_indx)
            remaining_values = pruned[remaining_indx]

            self._state = {
                "remaining_indx": np.where(remaining_indx),
                "remaining_values": remaining_values,
                "len": len(data),
            }

            print(f"Number of non-zeros: {number_of_non_zeros}")

            return pruned
        else:
            print(f"Number of non-zeros: {0}")
            self._state = {
                "remaining_indx": np.array([]),
                "remaining_values": np.array([]),
                "len": len(data),
            }
            return np.zeros_like(data)