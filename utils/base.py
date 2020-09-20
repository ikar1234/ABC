"""
Utility functions.
"""
import numpy as np

def encode_nucl(seq, seq_type='dna'):
    """
    Encode a DNA/RNA sequence for a time series distance.
    :param seq:
    :param seq_type:
    :return:
    """
    t = 'T'
    if seq_type == 'rna':
        t = 'U'
    code = {'A': 0, 'G': 1, 'C': 2, t: 3}
    res = np.array([code.get(i, -1) for i in seq])
    if -1 in res:
        raise ValueError("Unknown symbol detected.")
    return res
