"""
Approximate Bayesian Computation.
"""
import numpy as np
from math import ceil
from typing import Callable
from numpy import random
from ABC.distances import Distance
from warnings import warn
import json


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
                eval(f'random.{self.name}')
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

    def __call__(self, p):
        """
        Evaluate the distribution at a point.
        :param p: point in the domain of the distribution
        :return:
        """
        return self.func(p)


class Hierarchical:
    """
    # TODO: XML, more formats?
    Class for specifying hierarchical priors.
    Input schema is has the following form:
    A distribution is a dict-key and its value is again a dictionary with all parameters.
    Example: A Poisson-Gamma mixture, which is a Poisson distribution with parameter lambda
    being gamma-distributed with shape 1 and scale 2 looks as follows:
    h = {'poisson': {'lambda': {'gamma': {'shape':1, 'scale': 2} }}}
    Input can be either dict or a JSON object.
    """
    func: Callable

    def __init__(self, schema=None, schema_json=None):
        if schema_json is not None:
            # TODO: test
            schema = json.dumps(schema_json)

        while schema:
            ...

    def sample(self):
        ...


class ABCSampler:
    prior: Distribution

    def __init__(self, prior):
        """
        Initialize a sampler.
        :param prior: prior distribution, which could be sampled from
        """
        self.prior = prior

    def _rejection_sampler(self, model: dict, data, theta: str, size=100, dist_method='mmd', eps=0.01):
        """
        The simplest ABC algorithm - Rejection Sampling. First we sample some values from the prior and
        only accept those for which the simulated distribution are sufficiently close to the data.
        :param model: parameters of the model. This includes the name or custom function as well as
            the parameters of the function itself.
            Example: {'name': 'normal', 'params': {'loc': 1, 'scale': 2}}
        :param data: observed data
        :param theta: name of the parameter to be inferred
        :param size: number of samples taken from the prior
        :param eps: threshold for proximity of datasets
        :param dist_method: distance method for the data
        :return:
        """
        d = data.shape[0]
        dist = Distance(method=dist_method)
        thetas = self.prior.sample(size=size)
        for t in thetas:
            # add theta to the parameters
            model.setdefault(theta, t)
            l = Distribution.from_flat_dict(model)
            new_data = l.sample(size=d)
            if dist.compute(data, new_data) < eps:
                yield t

    def _mcmc_abc(self, model: dict, data, th0, theta: str, q=None, size=100, dist_method='mmd', eps=0.01):
        """
        The ABC-MCMC algorithm. We propose a value for the parameter. If the simulated sample is accepted,
        the next value is sampled with the MH rule, otherwise we keep the old value.
        :param model: parameters of the model. This includes the name or custom function as well as
            the parameters of the function itself.
            Example: {'name': 'normal', 'params': {'loc': 1, 'scale': 2}}
        :param data: observed data
        :param th0: initial value for the parameter
        :param theta: name of the parameter to be inferred
        :param q: sampling distribution for the parameter. default is standard normal
        :param size: number of samples taken from the prior
        :param dist_method: distance method for the data
        :param eps: threshold for proximity of datasets
        """
        if q is None:
            q = random.normal(loc=0, scale=1)

        for i in range(size):
            if type(q).__name__ == 'Distribution':
                # TODO: parameters
                th = q.sample(size=1)
            else:
                try:
                    # TODO: parameters
                    th = q(th0, size=1)
                except (ValueError, TypeError):
                    raise ValueError("Please provide a valid sampling distribution.")
            model.setdefault(theta, th)
            l = Distribution.from_flat_dict(model)
            # TODO
            new_data = l.sample(size=100)
            dist = Distance(method=dist_method)

            if dist.compute(data, new_data) < eps:
                # allow for a small numerical error
                if self.prior(th0) < 1e-15:
                    ...
                # TODO: q
                alpha = min(1, self.prior(th) / self.prior(th0))
                if alpha > random.uniform(low=0, high=1):
                    # accept new point
                    # TODO: multiple dimensions and call by reference
                    th0 = th
                    yield th0

    def _cma_abc(self, model: dict, data, th0: np.ndarray, theta: str, size=100, ratio=0.1, dist_method='mmd',
                 eps=0.01):
        """
        A CMA-version of the rejection ABC algorithm. If a value is accepted, then we sample new values which are
        close to the accepted one, and keep the best ones. This is similar to the CMA algorithm.
        The main distinction is the way we search for new values, namely by keeping the best ones and fitting a normal
        distribution to them.
        Warning! There are probably no theoretical guarantees for the convergence of the method.
        Since CMA is an optimization algorithm, I would assume that this, in a way, aims at maximizing the posterior.
        :param ratio: ratio of the sampled parameter which will be kept for the next iteration
        :return:
        """
        if ratio <= 0 or ratio > 1:
            raise ValueError('Ratio must be positive and smaller than 1.')

        n = th0.shape[0]
        loc = th0
        scale = np.eye(n)
        for i in range(size):
            thetas = random.normal(loc=loc, scale=scale, size=100)
            # array of distances
            dists = []
            for t in thetas:
                # add theta to the parameters
                model.setdefault(theta, t)
                l = Distribution.from_flat_dict(model)
                new_data = l.sample(size=100)
                dist = Distance(method=dist_method)
                dists.append(dist.compute(data, new_data))
                if dist.compute(data, new_data) < eps:
                    yield t

            # sort parameters by distance, where better parameters lead to shorter distance
            # TODO: numpy array?
            thetas = [x for y, x in sorted(zip(dists, thetas))]
            # keep only the best parameters
            best_params = ceil(100 * ratio)
            thetas = thetas[:best_params]
            # reestimate the parameters of the sampling distribution
            loc = np.mean(thetas)
            scale = np.cov(thetas)

    def _smc_abc(self, model, theta: str, N: int, data, T: int, eps=0.01, dist_method="mmd"):
        """
        An ABC-SMC algorithm. See https://arxiv.org/pdf/1608.07606.pdf, Algorithm 1 for reference
        :param model:
        :param theta:
        :param N:
        :param data:
        :param T:
        :param eps:
        :param dist_method:
        :return:
        """
        theta_vec = np.zeros((N, T))
        w = np.zeros((N, T))
        sigmas = np.zeros((1, T))
        dist = Distance(method=dist_method)

        if N <= 0:
            raise ValueError("Need at least 1 iteration.")
        new_data = None
        th = 0

        # iteration t = 0
        for i in range(N):
            while new_data is not None or dist.compute(data, new_data) > eps:
                th = self.prior.sample(size=1)
                model.setdefault(theta, th)
                l = Distribution.from_flat_dict(model)
                new_data = l.sample(size=100)
            theta_vec[i, 0] = th
            w[i, 0] = 1 / N
        sigmas[0] = 2 * np.cov(theta_vec[:, 0])

        # iteration t > 0
        for t in range(T):
            for i in range(N):
                while dist.compute(data, new_data) > eps:
                    th = np.random.choice(a=theta_vec[:, t - 1], size=1, p=w[:, t - 1])
                    # pertub theta* by adding gaussian noise
                    th += np.random.multivariate_normal(mean=np.zeros((1, N)), cov=sigmas[t - 1])
                    model.setdefault(theta, th)
                    l = Distribution.from_flat_dict(model)
                    new_data = l.sample(size=100)
                theta_vec[i, t] = th
                kernel = ...
                w[i, t] = self.prior(th) / (w[:, t - 1] @ kernel)
            sigmas[t] = ...

    def _hmm_abc(self):
        """
        An ABC algorithm for HMMs. A set of unobserved parameters are sampled to produce a sequence. This sequence
        is compared to the observed data.
        :return:
        """
        ...
