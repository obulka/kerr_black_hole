from numba import njit
import numpy as np


def sqr_norm(vec):
    """ 
    """
    return np.einsum("...i,...i", vec, vec)
