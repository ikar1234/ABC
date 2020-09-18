"""
Approximate Bayesian Computation.
"""
from typing import Callable
from numpy import random


class Sampler:
    """
    A basic class used to sample from the prior.
    The preferred option is to have installed PyMC3.
    Supported methods are:
        - Inverse transform
        - Alias sampling
        - Rejection sampling
        - Importance sampling
        - Annealed importance sampling
        - MCMC (Metropolis-Hasting, Gibbs sampling)
        - Slice sampling

    Default is Metropolis-Hastings.
    """
    method: str
    params: dict

    def __init__(self, method="MH", params=None):
        ...

    def sample(self, func, size=1):
        ...


# TODO: rename to prior?
class Distribution:
    func: Callable
    name: str
    params: dict

    def __init__(self, name=None, func=None, params=None):
        self.func = func
        self.name = name
        if self.name is not None:
            try:
                f = eval(f'random.{self.name}')
            except AttributeError:
                raise ValueError('Distribution not found.')
        if params is None:
            self.params = dict()
        self.params = params

    def sample(self, size=1, method='mcmc'):
        if self.name is not None:
            try:
                f = eval(f'random.{self.name}')
                return f(size=size, **self.params)
            except AttributeError:
                raise ValueError('Distribution not found.')
        elif self.func is not None:
            s = Sampler(method=method, params=self.params)
            return s.sample(self.func, size=size)
        else:
            raise ValueError('Function to sample from was not provided.')


# TODO: Allow for hierarchical distributions by specifying a JSON format. Example
#       h = {'poisson': {'lambda': {'gamma': {'shape':1, 'scale': 2} }}]}
class Hierarchical:
    schema: dict

    def __init__(self, schema):
        ...


class ABCSampler:
    prior: Callable

    def __init__(self, prior):
        """
        Initialize a sampler.
        :param prior: prior distribution, which could be sampled from
        """
        self.prior = prior

    def fit(self, likelihood, data):
        """

        :param likelihood:
        :param data: observed data
        :return:
        """
