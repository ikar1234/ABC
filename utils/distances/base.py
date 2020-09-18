"""
Distance functions for datasets.
Does not include distances that are used in hypothesis tests and are already implemented
in some packages.
"""


def mmd(d1, d2):
    """
    Maximum Mean Discrepancy.
    :return:
    """
    # TODO
    return sum([x in d2 for x in d1]) + sum([x in d1 for x in d2])


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
