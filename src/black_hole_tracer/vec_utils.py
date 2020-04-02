from numba import njit
import numpy as np


def vec3a(vec): # returns a constant 3-vector array (don't use for varying vectors)
    return np.outer(ones,vec)


def vec3(x,y,z):
    return vec3a(np.array([x,y,z]))


def norm(vec):
    return np.sqrt(sqr_norm(vec))


def normalize(vec):
    return vec / (norm(vec)[:, np.newaxis])


def sqr_norm(vec):
    """ Efficient way of computing the sixth power of r, much faster
    than pow. np has this optimization for power(a,2) but not for
    power(a,3)
    """
    return np.einsum("...i,...i", vec, vec)


def sixth(v):
    tmp = sqr_norm(v)
    return tmp * tmp * tmp


#for shared arrays
def to_numpy_array(mp_arr, num_pixels):
    a = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
    a.shape = (num_pixels, 3)
    return a
