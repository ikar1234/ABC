import cProfile

from ABC.distances import TSDistance


def hamming_loc(d1, d2):
    l = zip(d1, d2)
    return sum([x[0] == x[1] for x in l])

def profile_tsdistance():
    data1 = b"ACGACGACAACTAC" * 100000
    data2 = b"ACTACGTCAGATAG" * 100000

    d = TSDistance(method='hamming')
    cProfile.runctx('d.compute(data1, data2)', globals(), locals())


if __name__ == '__main__':
    profile_tsdistance()
