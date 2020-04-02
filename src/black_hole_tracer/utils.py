import numpy as np


def vec3a(vec): # returns a constant 3-vector array (don't use for varying vectors)
    return np.outer(ones,vec)


def vec3(x,y,z):
    return vec3a(np.array([x,y,z]))


def norm(vec):
    return np.sqrt(sqrnorm(vec))


def normalize(vec):
    return vec / (norm(vec)[:, np.newaxis])


def sqrnorm(vec):
    """ Efficient way of computing the sixth power of r, much faster
    than pow. np has this optimization for power(a,2) but not for
    power(a,3)
    """
    return np.einsum("...i,...i", vec, vec)


def sixth(v):
    tmp = sqrnorm(v)
    return tmp * tmp * tmp


def RK4f(y, h2):
    f = np.zeros(y.shape)
    f[:, 0:3] = y[:, 3:6]
    f[:, 3:6] = (
        -1.5 * h2 * y[:, 0:3]
        / np.power(sqrnorm(y[:, 0:3]), 2.5)[:, np.newaxis]
    )
    return f


#for shared arrays
def tonumpyarray(mp_arr):
    a = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
    a.shape = ((num_pixels, 3))
    return a