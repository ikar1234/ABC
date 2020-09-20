"""
Distance functions for datasets.
Does not include distances that are used in hypothesis tests and are already implemented
in some packages.
"""
import numpy as np
from importlib.util import find_spec


def mean_diff(d1, d2):
    """
    Difference of means. Useful for high-dimensional data, but it can underestimate the true distance since
    it does not differentiate distributions with different higher moments.
    :param d1: first dataset
    :param d2: second dataset
    :return:
    """
    return np.abs(np.mean(d1) - np.mean(d2))


def mmd(d1, d2):
    """
    Maximum Mean Discrepancy.
    :return:
    """
    # TODO


def kmmd(d1, d2, k, alpha: float):
    """
    Kernel MMD.
    :param d1: first dataset
    :param d2: second dataset
    :param k: positive definite kernel
    :param alpha: level of test

    See doi:10.1093/bioinformatics/btl242 for reference.
    :return:
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('Parameter alpha must lie between 0 and 1.')
    if d1.shape != d2.shape:
        raise ValueError("Input datasets are not of equal shape.")


# TODO
class Distance:
    method: str

    def __init__(self, method):
        self.method = method

    def compute(self, data1, data2):
        if self.method == 'mmd':
            return mmd(data1, data2)
        elif self.method == 'mean_diff':
            return mean_diff(data1, data2)


class TSDistance(Distance):
    """
    Distance for time-series observations.
    Includes the following methods:
    - Hamming distance
    - Edit distance
    - Dynamic Time Warping (DTW)
    """
    method: str

    def compute(self, data1, data2):
        if find_spec('tslearn'):
            return self._compute_tslearn(data1, data2)
        else:
            return self._compute_native(data1, data2)

    def _compute_tslearn(self, data1, data2):
        from tslearn.utils import to_time_series
        from tslearn.metrics import dtw
        ts1 = to_time_series(data1)
        ts2 = to_time_series(data2)

        if self.method == 'dtw':
            return dtw(ts1, ts2)
        return 3.14

    def _compute_native(self, data1, data2):
        return 2.71
