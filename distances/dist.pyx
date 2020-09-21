# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from typing import Union

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

cpdef int hamming(char *d1, char *d2):
    l = zip(d1, d2)

    cdef Py_ssize_t c
    cdef (char, char) i
    c = 0
    while 1:
        i = next(l, None)
        if i is not None:
            if i[0] == i[1]:
                c += 1
        else:
            break
    return c

def norm_hamming(d1, d2):
    return hamming(d1, d2) / len(d1)

def cumsum_ts(d1, d2, p: Union[int, str, None] = None):
    """
    Cumulative distance for time series. First, the time series sequences are cumulated and the resulting
    sequences are compared using the standard Euclidean metric.
    :param d1: first dataset
    :param d2: second dataset
    :param p: the metric parameter
    :return:
    """
    ts1 = np.cumsum(d1)
    ts2 = np.cumsum(d2)

    return np.linalg.norm(ts1, ts2, p)
