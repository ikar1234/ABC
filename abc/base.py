"""
Approximate Bayesian Computation.
"""
from typing import Callable
from numpy import random
from ABC.utils.distances import Distance
from warnings import warn


class Sampler:
    """
    A basic class used to sample from the prior.
    The preferred option is to have installed PyMC3.
    Supported methods are:
        - Alias sampling /for discrete distributions/ (alias)
            Parameters: none
        - Rejection sampling (rej)
            Parameters: q: Second distribution. If not specified, (an upper bound of) the maximum is taken
        - Importance sampling resampling (sir)
            Parameters: q: Second distribution.
        - Annealed importance sampling (ann-imp)
            Parameters: none
        - MCMC (Metropolis-Hasting, Gibbs sampling): (mh), (gibbs)
            Parameters: q: Proposal distribution. Default is a standard normal (RW-Metropolis)
        - Slice sampling (slice)
            Parameters: none

    Default is Metropolis-Hastings.
    """
    method: str
    params: dict

    def __init__(self, method="mh", params=None):
        self.method = method
        if params is None:
            self.params = dict()
        else:
            self.params = params

    def sample(self, func, size=1):
        if self.method.lower() == 'alias':
            ...
        elif self.method.lower() == 'rej':
            ...
        elif self.method.lower() == 'sir':
            ...
        elif self.method.lower() == 'ann-imp':
            ...
        elif self.method.lower() == 'mh':
            return [11 for _ in range(size)]
        elif self.method.lower() == 'gibbs':
            ...
        elif self.method.lower() == 'slice':
            ...
        else:
            raise ValueError("Not a valid method.")


# TODO: rename to prior?
class Distribution:
    """
    A probability distribution class, specified either by its name, which must be the same as the name
    of the corresponding numpy.random distribution, or by a valid density function. In the latter case,
    samples from the function will be only approximate.
    """
    func: Callable
    name: str
    params: dict

    def __init__(self, name=None, func=None, params=None):
        """
        Initialize the distribution.
        :param name: name of the distribution, which must coincide with the name of a numpy.random distribution
        :param func: custom function, which will be approximately sampled from
        :param params: parameters of the distribution. Can be used for both types of distributions
        """
        self.func = func
        self.name = name
        if self.name is not None:
            if self.func is not None:
                warn(message='Both name and a custom function were provided. I wont use the custom function.')
            try:
                f = eval(f'random.{self.name}')
            except AttributeError:
                raise ValueError('Distribution not found.')
        elif self.func is None:
            raise ValueError('You must provide either the distribution name or a custom function.')
        if params is None:
            self.params = dict()
        else:
            self.params = params

    @classmethod
    def from_flat_dict(cls, params: dict):
        """
        Construct the distribution from a flat dictionary, i.e. the name/func parameter
        is included in the params dict. Main use case is construction of the likelihood model.
        :param params: dictionary. Contains the name or func parameter as well as all
            other distribution-specific parameters.
        :return:
        """
        params_copy = params.copy()
        name = params_copy.get('name', None)
        func = params_copy.get('func', None)

        if name is None and func is None:
            raise ValueError('You must provide either the distribution name or a custom function.')
        # remove from the parameters of the distribution
        params_copy.pop('name', None)
        params_copy.pop('func', None)
        return cls(name=name, func=func, params=params_copy)

    def sample(self, size=1, method='mh'):
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


class Hierarchical:
    """
    # TODO: XML, more formats?
    Class for specifying hierarchical priors.
    Input schema is has the following form:
    A distribution is a dict-key and its value is again a dictionary with all parameters.
    Example: A Poisson-Gamma mixture, which is a Poisson distribution with parameter lambda
    being gamma-distributed with shape 1 and scale 2 looks as follows:
    h = {'poisson': {'lambda': {'gamma': {'shape':1, 'scale': 2} }}}
    """
    schema: dict

    def __init__(self, schema):
        while schema:
            ...


class ABCSampler:
    prior: Distribution

    def __init__(self, prior):
        """
        Initialize a sampler.
        :param prior: prior distribution, which could be sampled from
        """
        self.prior = prior

    def fit(self, model: dict, data, theta: str, size=100, method='mmd', eps=0.01):
        """

        :param model: parameters of the model. This includes the name or custom function as well as
            the parameters of the function itself.
            Example: {'name': 'normal', 'params': {'loc': 1, 'scale': 2}}
        :param data: observed data
        :param theta: name of the parameter to be inferred
        :param size: number of samples taken from the prior
        :param eps: threshold for proximity of datasets
        :param method: distance method for the data
        :return:
        """
        d = data.shape[0]
        dist = Distance(method=method)
        thetas = self.prior.sample(size=size)
        for t in thetas:
            # add theta to the parameters
            model.setdefault(theta, t)
            l = Distribution.from_flat_dict(model)
            new_data = l.sample(size=d)
            if dist.compute(data, new_data) < eps:
                yield t
