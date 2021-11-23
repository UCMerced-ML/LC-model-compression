import numpy as np

"""
    TODO: This file is poorly documented for now, will fix in future version.
"""
def diff_based_coding(dense_array_of_diffs, correction_precision=16, diff_bases=range(3, 20)):
    min_total_diff_cost = float('inf')
    for diff_base in diff_bases:
        max_diff = 2**diff_base-1
        total_diffs = 0
        # there is a high chance that the line below is meaningles: i.e., will never change the data, i.e
        # sparse_array_of_diffs is always equal to dense_array_of_diffs. TODO: check later
        sparse_array_of_diffs = dense_array_of_diffs[dense_array_of_diffs != -1]
        for diff in sparse_array_of_diffs:
            if diff < max_diff:
                total_diffs += 1
            else:
                total_diffs += (diff//max_diff + 1)
        if total_diffs*diff_base+len(dense_array_of_diffs)*correction_precision < min_total_diff_cost:
            min_total_diff_cost = total_diffs*diff_base+len(dense_array_of_diffs)*correction_precision

    return min_total_diff_cost
