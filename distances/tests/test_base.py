import unittest

from ABC.distances import TSDistance
from ABC.utils import encode_nucl


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_tslearn(self):
        data1 = encode_nucl("ACGACGACAACTAC")
        data2 = encode_nucl("ACTACGTCAGATAG")

        d = TSDistance(method='dtw')
        print(d.compute(data1, data2))


if __name__ == '__main__':
    unittest.main()
