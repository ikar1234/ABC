"""
Distance functions for datasets.
Does not include distances that are used in hypothesis tests and are already implemented
in some packages.
"""

import numpy as np
from importlib.util import find_spec
from .dist import mean_diff, mmd, kmmd, hamming, cumsum_ts

distances = [mean_diff, mmd, kmmd]

ts_distances = [hamming, cumsum_ts]


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
    It is preferred to have the tslearn package installed, which provides efficient
    implementations of most methods.
    Nevertheless, the native methods are implemented in Cython and thus also scale well.
    Includes the following methods:
    - Hamming distance
    - Cumulative Euclidean distance
    - All distances provided by tslearn (given that it is installed)
    """
    method: str

    def compute(self, data1, data2, tslearn_dist=False, **params):
        if tslearn_dist and find_spec('tslearn'):
            return self._compute_tslearn(data1, data2, **params)
        else:
            return self._compute_native(data1, data2, **params)

    def _compute_tslearn(self, data1, data2, **params):
        """
        Use a distance function from the tslearn package.
        :param data1:
        :param data2:
        :param params:
        :return:
        """
        from tslearn.utils import to_time_series
        import tslearn.metrics
        ts1 = to_time_series(data1)
        ts2 = to_time_series(data2)

        try:
            f = eval(f'tslearn.metrics.{self.method}')
            return f(ts1, ts2, params)
        except AttributeError:
            raise ValueError("Method not found.")

    def _compute_native(self, data1, data2, **params):
        """
        Use a self-implemented distance function.
        :param d1: first dataset
        :param d2: second dataset
        :param params:
        :return:
        """
        funcs = [f for f in ts_distances if self.method == f.__name__]
        if len(funcs) > 0:
            f = funcs[0]
            return f(data1, data2, **params)
        return
