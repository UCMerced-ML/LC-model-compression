#!/usr/bin/env python3
from .base_types import CompressionTypeBase
from sklearn.cluster import KMeans
import numpy as np
import warnings
"""
References:
[1] Miguel Á. Carreira-Perpiñán and Yerlan Idelbayev
    Model compression as constrained optimization, with application to neural nets. Part II: quantization
    https://arxiv.org/abs/1707.04319
[2] David Arthur and Sergei Vassilvitskii
    k-means++:  The Advantages of Careful Seeding
    http://dl.acm.org/citation.cfm?id=1283494
[3] Xiaolin Wu and John Rokne
    An O(KN logN) Algorithm for Optimum K-level Quantization on Histograms of N Points
    http://doi.acm.org/10.1145/75427.75472
"""
class AdaptiveQuantization(CompressionTypeBase):
    def __init__(self, k):
        """
        This function constructs an object that returns an approximate solution of the adaptive quantization problem.
        Given a vector, it would return same-sized vector such that every item is coming from the codebook of size k,
        and overall quadratic distortion between quantized and original vectors are minimized.

        The solution is obtained by first running the k-means++[2] to get better initial guess and then by performing
        simple k-means iterations. If self.compress() called subsequently, we expect the new codebook entries of the
        solution are being not far away from previous one, therefore for next compress() calls, we warmstart from
        previous solution.

        See [1] for more details.
        # TODO: add page numbers
        :param k: the size of the codebook
        """
        super(AdaptiveQuantization, self).__init__()

        self.k_ = k

    def load_state_dict(self, state_dict):
        self._state = state_dict

    def uncompress_state(self):
        cluster_centers = self.state_dict['cluster_centers']
        assignments = self.state_dict['assignments']
        if self.k_ == 2:
            # unpack binary-bits if necessary
            assignments = np.unpackbits(assignments)
            if 'data_shape' in self.state_dict:
                l = np.prod(self.state_dict['data_shape'])
                assignments = assignments[:l]
            else:
                print('OLD FORMAT: ++ =( +++')
        quantized = cluster_centers.flatten()[assignments]
        return quantized

    def compress(self, data):
        with warnings.catch_warnings():
            # disabling these nasty initial center position warnings
            warnings.simplefilter("ignore")
            if self.state_dict is not None:
                init = self.state_dict["cluster_centers"]
            else:
                init = 'k-means++'
            print(self._state, init)
            kmeans = KMeans(n_clusters=self.k_, n_init=10, tol=1e-10, init=init)
            assignments = kmeans.fit_predict(data[:, None])
            print('K-Means converged in {} iterations.'.format(kmeans.n_iter_))
            if self.step_number not in self.info:
                self.info[self.step_number] = []
            cluster_centers = kmeans.cluster_centers_
            self.info[self.step_number].append({"cluster_centers": cluster_centers, "iterations_to_converge": kmeans.n_iter_})

            self._state = {"cluster_centers": cluster_centers, "assignments": assignments, "data_shape": data.shape}

            if self.k_ == 2:
                # we can store much more efficiently
                self._state['assignments'] = np.packbits(assignments)

            quantized = cluster_centers.flatten()[assignments]
            return quantized


class OptimalAdaptiveQuantization(CompressionTypeBase):
    """
    This class solves k-means optimally for 1d quantization case using K*N*log(N) algorithm of Wu and Rockne, see [3].
    """
    def __init__(self, k):
        self.k_ = k

    def optimal_kmeans(self, xs):
        if np.ndim(xs) != 1:
            raise NotImplementedError("DP KMeans only works with 1d arrays.")

        xs = np.sort(xs, kind='mergesort')
        # assume no duplicates
        if len(xs) <= self.k_:
            return 0, xs

        for_cum_sum = np.zeros(len(xs) + 1)
        for_cum_sum[1:] = xs
        forward_running_sum = np.cumsum(for_cum_sum)
        forward_running_sum_sq = np.cumsum(for_cum_sum ** 2)

        def mean_var(i, j):
            """computes mean and var between [x_i, x_(i+1), ..., x_(j-1)] same as [i:j]"""
            if j <= i:
                raise Exception("j should be always > i")
            if i + 1 == j:
                return xs[i], 0
            mean = (forward_running_sum[j] - forward_running_sum[i]) / (j - i)
            sum_sq = (forward_running_sum_sq[j] - forward_running_sum_sq[i])
            var = sum_sq - (j - i) * (mean ** 2)
            if var < 0:
                if abs(var) < 10**(-14):
                    return mean, 0
                else:
                    # number of such cases must be extremely small, so no worries
                    print(f"recomputing the variance on {i},{j}")
                    return mean, np.var(xs[i:j])
            return mean, var

        DP = {}

        def find_opt(l, r, k, n):
            "finds DP(n,k) searching from x_l to x_r"
            if l >= r:
                mean, val = mean_var(l, n + 1)
                val_, sol_, index = DP[(n, k - 1)]
                DP[(n, k)] = (val + val_, sol_ + [mean], l)
                return
            min_so_far = float('Inf')
            for j in range(max(l, k - 2), r):
                mean, var = mean_var(j + 1, n + 1)
                if DP[(j, k - 1)][0] + var < 0:

                    raise Exception("DP[(j, k - 1)][0] + var < 0, var={}, DP={}, j+1={}, n+1={}".format(var, DP[(j, k - 1)][0], j+1, n+1))
                if DP[(j, k - 1)][0] + var < min_so_far:
                    min_so_far = DP[(j, k - 1)][0] + var
                    DP[(n, k)] = (min_so_far, DP[(j, k - 1)][1] + [mean], j)

        def find_all_opt(l, r, k):
            if r <= l:
                return
            mid = (l + r) // 2

            start = max(DP[(mid, k - 1)][2], DP[(l - 1, k)][2])
            end = min(mid, DP[(r, k)][2] + 1)
            find_opt(start, end, k, mid)
            find_all_opt(l, mid, k)
            find_all_opt(mid + 1, r, k)

        for i in range(len(xs)):
            mean, var = mean_var(0, i + 1)
            DP[(i, 1)] = (var, [mean], 0)

        for kk in range(2, self.k_ + 1):
            DP[(kk - 1, kk)] = (0, xs[:kk], 0)

            start = 0
            start = max(start, DP[(len(xs) - 1, kk - 1)][2])

            find_opt(start, len(xs) - 1, kk, len(xs) - 1)
            if kk != self.k_:
                left = kk
                right = len(xs) - 1
                find_all_opt(left, right, kk)

        return DP[(len(xs) - 1, self.k_)][:2]

    def quantize_to(self, data, centers):
        quantized = np.zeros(len(data))
        for i, x in enumerate(data):
            min_ = float("Inf")
            for c in centers:
                val = (x-c)**2
                if val < min_:
                    quantized[i] = c
                    min_ = val
        return quantized

    def compress(self, data):
        error, centers = self.optimal_kmeans(data)
        quantized = self.quantize_to(data, centers)
        print('error is ', error)
        return quantized


class ScaledBinaryQuantization(CompressionTypeBase):
    def __init__(self):
        """
        This function constructs an object that returns an exact solution of the binary quantization problem,
        where codebook entries are of [-a, a], and a is learned.

        See [1] for more details.
        """
        super(ScaledBinaryQuantization, self).__init__()

    def compress(self, data):
        a = np.mean(np.abs(data))
        quantized = 2 * a * (data > 0) - a

        return quantized


class BinaryQuantization(CompressionTypeBase):
    def __init__(self):
        """
        This function constructs an object that returns an exact solution of the binary quantization problem,
        where codebook entries are of [-1, 1].

        See [1] for more details.
        """
        super(BinaryQuantization, self).__init__()

    def compress(self, data):
        a = 1
        quantized = 2 * a * (data > 0) - a

        return quantized


class ScaledTernaryQuantization(CompressionTypeBase):
    def __init__(self):
        """
        This function constructs an object that returns an exact solution of the binary quantization problem,
        where codebook entries are of [-a, 0, a], and a is learned.

        See [1] for more details.
        """
        super(ScaledTernaryQuantization, self).__init__()

    def compress(self, data):
        # We need a stable sort so that same valued weights will not get assigned to different
        # clusters. The only stable sort in numpy is MergeSort.
        sorted_x = np.sort(np.abs(data), kind='mergesort')[::-1]
        cumsum = 0
        max_so_far = float('-inf')
        max_indx = -1
        for i, x_ in enumerate(sorted_x):
            cumsum += x_
            j_i = cumsum / (i + 1) ** 0.5
            if j_i > max_so_far:
                max_so_far = j_i
                max_indx = i
        a = np.mean(sorted_x[:max_indx + 1])
        _abs = np.abs(data)
        _nonzero = data * [_abs > a / 2]
        quantized = np.sign(_nonzero) * a

        return quantized
