#!/usr/bin/env python3
from .base_types import CompressionTypeBase
import numpy as np
from scipy.linalg import svd

"""
References:
[1] Yerlan Idelbayev and Miguel Á. Carreira-Perpiñán
    Low-rank Compression of Neural Nets: Learning the Rank of Each Layer, CVPR2020,
    http://graduatestudent.ucmerced.edu/yidelbayev/papers/cvpr20/cvpr20a.pdf
"""

def tensor_to_matrix(tensor, conv_scheme):
    """
    Reshape a tensor into a matrix according to a scheme.
    :param tensor:
    :param conv_scheme: currently we support two scheme: scheme-1 and scheme-2
    :return: a matrix containing tensor's elements
    """
    matrix = tensor
    init_shape = tensor.shape
    if tensor.ndim == 4:
        if conv_scheme == 'scheme_1':
            matrix = tensor.reshape(init_shape[0], -1)
        elif conv_scheme == 'scheme_2':
            [n, m, d1, d2] = tensor.shape
            swapped = tensor.swapaxes(1, 2)
            matrix = swapped.reshape([n * d1, m * d2])
        else:
            raise NotImplementedError('This type of shape parametrization is not implemented yet.')

    if not np.allclose(tensor, matrix_to_tensor(matrix, init_shape, conv_scheme)):
        raise NotImplementedError("The Tensor->Matrix->Tensor conversion is not correctly implemented.")
    return matrix


def matrix_to_tensor(matrix, init_shape, conv_scheme):
    """
    Reshapes previously reshaped matrix into the original tensor.
    :param matrix: matrix to be converted back
    :param init_shape: the initial shape of the tensor
    :param conv_scheme: the convolutional scheme used for reshape
    :return:
    """
    tensor = matrix
    if len(init_shape) == 4:
        if conv_scheme == "scheme_1":
            tensor = matrix.reshape(init_shape)
        elif conv_scheme == 'scheme_2':
            [n, m, d1, d2] = init_shape
            tensor = matrix.reshape([n, d1, m, d2]).swapaxes(1, 2)
        else:
            raise NotImplementedError('This type of shape parametrization is not implemented yet.')
    return tensor


class LowRank(CompressionTypeBase):
    def __init__(self, target_rank, conv_scheme=None, precision='float'):
        """
        The low-rank compression of the matrix into given low-rank. If tensor is given, then it should be reshaped into
        a matrix according to a conv_scheme: either of scheme_1 or scheme_2.

        :param target_rank: the target rank of compressed weights
        :param conv_scheme: if tensor is given, how it should be reshaped into matrix
        :param precision: should be store low-rank U,V with different precision (32 or 16 bits)
        """
#         print("are we here?")
        super(LowRank, self).__init__()
        self.target_rank = target_rank
        self.conv_scheme = conv_scheme
        self.precision = (32 if precision == 'float' else precision)

    def perform_svd(self, matrix):
        u, s, v = svd(matrix, full_matrices=False)
        return u,s,v

    def compress(self, data):
        print("we are here")
        matrix = None
        init_shape = data.shape
        if data.ndim == 2:
            matrix = data
        elif data.ndim == 4:
            matrix = tensor_to_matrix(data, self.conv_scheme)

        u, s, v = self.perform_svd(matrix)
        r = self.target_rank
        print("target_rank", self.target_rank)
        if r < np.min(matrix.shape):
            diag = np.diag(s[:r] ** 0.5)
            U = u[:, :r] @ diag
            V = diag @ v[:r, :]

            if self.precision == 16:
                print("Storing U,V matrices as 16bit IEEE floating point matrices.")
                U = U.astype(np.float16)
                V = V.astype(np.float16)

            low_rank_matrix = U @ V
            self.info[self.step_number] = [U, V]

            step_info = {"selected_rank": None, "singular_values": s }
            self.info[self.step_number] = step_info
            self._state = {"U": U, "V": V, "selected_rank": None, "init_shape": init_shape}

            print("Low-rank compression is finished")
            if len(init_shape) == 2:
                return low_rank_matrix
            elif len(init_shape) == 4:
                return matrix_to_tensor(low_rank_matrix, init_shape, self.conv_scheme)
        
        

    def load_state_dict(self, state_dict):
        self._state = state_dict

    def uncompress_state(self):
        U = self.state_dict['U']
        V = self.state_dict['V']
        if len(self.state_dict["init_shape"]) == 2:
            return U@V
        elif len(self.state_dict["init_shape"]) == 4:
            return matrix_to_tensor(U@V, self.state_dict["init_shape"], self.conv_scheme)


class RankSelection(LowRank):
    def __init__(self, conv_scheme, alpha, criterion, normalize, module):
        """
        This is the C step corresponding to optimization of:
            min_{U,V,r} ‖w-UVᵀ‖² + 𝛼 C(r)  s.t.  rank(w) <= r
        Which allows to automatically learn both rank and weights for low-rank compression of a neural network
        according to model selection criterion C(r): storage compression or FLOPs compression.

        See [1] for full details.

        :param conv_scheme: if compressed weights are in tensor form, then we reshape them according to a particular
            scheme: "scheme_1" or "scheme_2". If weights are in matrix form, this parameter is ignored.
        :param alpha: the trade-off parameter controlling the amount of compression, higher the alpha, higher the
            the compression. In the paper [1] we refer to it as lambda. Also, in the paper, the values of lambda is
            given wrt millions of parameters or flops, thus rescaled by 10^6. However in code, we don't use rescaled
            lambda, thus call it alpha. For example, if in the paper we give lambda=10^{-4}, you need to use
            alpha=10^{-10}.
        :param criterion: the selection criterion, either "storage" or "flopgs".
        :param normalize: whether we normalize the C-step loss by 1/‖w‖² or not. We find it useful for deep nets.
        :param module: the PyTorch module (layer) storing the weights, used to obtain the flops information.
        """
        super(RankSelection, self).__init__(None, conv_scheme)
        self.alpha = alpha
        self.module = module
        self.criterion = criterion
        self.selected_rank = 0
        self.normalize_matrix = normalize

    def compress(self, data):
        init_shape = data.shape
        if data.ndim == 2:
            matrix = data
        elif data.ndim == 4:
            matrix = tensor_to_matrix(data, self.conv_scheme)

        u, s, v = self.perform_svd(matrix)
        m, n = matrix.shape
        max_rank = min(matrix.shape)
        selected_rank = 0
        best = float('Inf')
        rank_i_diff_frb_norm_sq = (s[::-1] ** 2).cumsum()[::-1]
        rank_i_frb_norm_sq = (s[:] ** 2).cumsum()

        norm_sq = 1.0
        mult = 1 if self.criterion == 'storage' else self.module.multiplier

        def cost(r):
            return r*(m+n)*mult


        if self.normalize_matrix:
            frob_norm_sq = rank_i_diff_frb_norm_sq[0]
            norm_sq = frob_norm_sq

        suggested_mu = 2 * self.alpha * (m + n) * mult / (s[0] ** 2 / norm_sq)
        # print(f'With current α={self.alpha:.2e}, μ should be at least {suggested_mu:.2e}')

        for r in range(0, max_rank + 1):
            value = self.alpha * cost(r) \
                    + ((self.mu / 2) * rank_i_diff_frb_norm_sq[r]*(1/norm_sq) if r < max_rank else 0)
            # equivalent to =>
            # + self.mu/2*np.sum((values_to_compress-new_weights)**2)

            if value < best:
                selected_rank = r
                best = value

        diag = np.diag((s[:selected_rank]) ** 0.5)
        U = u[:, :selected_rank] @ diag
        V = diag @ v[:selected_rank, :]
        low_rank_matrix = U @ V

        if selected_rank <= max_rank:
            frob_norm_diff = (rank_i_diff_frb_norm_sq[selected_rank]/norm_sq if selected_rank < max_rank else 0)
        else:
            frob_norm_diff = 0

        self.selected_rank = selected_rank
        step_info = {
            "selected_rank": selected_rank,
            "singular_values": s,
            "optimal_loss": best,
            "suggested_mu": suggested_mu
        }
        print(f"for this layer, selected rank is {selected_rank}, "
              f"normalized ‖w‖²={(s ** 2).sum() / norm_sq:.3f}, "
              f"true ‖w‖²={(s ** 2).sum():.3f}, "
              f"‖Δ(Θ)‖²={(low_rank_matrix ** 2).sum() / norm_sq:.3f}, "
              f"‖w-Δ(Θ)‖²={frob_norm_diff:.3e}, "
              f"μ should be at least {suggested_mu:.2e}")
        self.info[self.step_number] = step_info
        self._state = {"U": U, "V": V, "selected_rank": selected_rank, "init_shape": init_shape}

        if len(init_shape) == 2:
            return low_rank_matrix
        elif len(init_shape) == 4:
            return matrix_to_tensor(low_rank_matrix, init_shape, self.conv_scheme)