import logging
import matplotlib.pyplot as plt
import numpy as np


def convert_image_to_float(image):
    """
    """
    return image.astype(float) / 255.


def rgbtosrgb(arr):
    """ Convert from linear rgb to srgb. """
    # See https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    mask = arr > 0.0031308
    arr[mask] **= 1/2.4
    arr[mask] *= 1.055
    arr[mask] -= 0.055
    arr[np.logical_not(mask)] *= 12.92


def srgbtorgb(arr):
    """ Convert from srgb to linear rgb. """
    mask = arr > 0.04045
    arr[mask] += 0.055
    arr[mask] /= 1.055
    arr[mask] **= 2.4
    arr[np.logical_not(mask)] /= 12.92


def lookup(texture, uv_arr_in):
    """ Perform texture lookup.

    Args:
        texture (np.array): Array containing texture.
        uv_arr_in (np.array): Array of uv coordinates
    """
    uv_array = np.clip(uv_arr_in, 0.0, 0.999)

    uv_array[:, 0] *= float(texture.shape[1])
    uv_array[:, 1] *= float(texture.shape[0])

    uv_array = np.round(uv_array).astype(int)

    return texture[uv_array[:, 1], uv_array[:, 0]]


def save_to_img(arr, file_name, resolution, srgb_out=True):
    """
    """
    img_out = np.array(arr)
    img_out = np.clip(img_out, 0.0, 1.0)

    if srgb_out:
        rgbtosrgb(img_out)

    img_out = img_out.reshape((resolution[1], resolution[0], 3))

    plt.imsave(file_name, img_out)


def post_process(image, gain, normalize):
    """
    """
    image *= gain

    # Normalization
    if normalize > 0:
        max_pixel = np.amax(image.flatten())
        if max_pixel:
            image *= 1 / (normalize * max_pixel)

    return np.clip(image, 0., 1.)
