import unittest
import numpy as np
from numpy.random import normal

from ABC.abc.base import Distribution, ABCSampler


class MyTestCase(unittest.TestCase):
    def test_distribution(self):
        a = 10
        b = 12
        d = Distribution(name='uniform', params={'low': a, 'high': b})
        s = d.sample(size=10)
        for i in s:
            # assert samples are in the specified bounds
            self.assertGreaterEqual(i, a)
            self.assertLessEqual(i, b)

    def test_custom_distribution(self):
        a = 10
        b = 12
        f = lambda x: 1 / (b - a)
        d = Distribution(func=f)
        s = d.sample(size=10)
        for i in s:
            # assert samples are in the specified bounds
            self.assertGreaterEqual(i, a)
            self.assertLessEqual(i, b)

    def test_abc_1d(self):
        # data follows a N(5,1) distribution
        # we aim to find the mean, while the variance is assumed to be known
        data = normal(loc=5, scale=1, size=50)
        # prior mean is Gaussian, with its peak being close to the true parameter
        prior = Distribution(name='normal', params={'loc': 3, 'scale': 2})
        # likelihood model
        # we know that the data is normal and are trying to infer the mean
        lkh_model = {'name': 'normal', 'scale': 1}
        abc = ABCSampler(prior=prior)
        thetas = abc._rejection_sampler(data=data, theta='loc', model=lkh_model, dist_method='mean_diff', eps=2)
        print([x for x in thetas])
        print(np.mean(list(thetas)))


if __name__ == '__main__':
    unittest.main()
