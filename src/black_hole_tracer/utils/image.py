import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def convert_image_to_float(image):
    """
    """
    return image.astype(float) / 255.


# convert from linear rgb to srgb
def rgbtosrgb(arr):
    logger.debug("RGB -> sRGB...")
    #see https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    mask = arr > 0.0031308
    arr[mask] **= 1/2.4
    arr[mask] *= 1.055
    arr[mask] -= 0.055
    arr[np.logical_not(mask)] *= 12.92


# convert from srgb to linear rgb
def srgbtorgb(arr):
    logger.debug("sRGB -> RGB...")
    mask = arr > 0.04045
    arr[mask] += 0.055
    arr[mask] /= 1.055
    arr[mask] **= 2.4
    arr[np.logical_not(mask)] /= 12.92


#defining texture lookup
def lookup(texarr, uv_arr_in): # uv_arrayin is an array of uv coordinates
    uv_array = np.clip(uv_arr_in, 0.0, 0.999)

    uv_array[:, 0] *= float(texarr.shape[1])
    uv_array[:, 1] *= float(texarr.shape[0])

    uv_array = uv_array.astype(int)

    return texarr[uv_array[:, 1], uv_array[:, 0]]


def save_to_img(arr, file_name, resolution, srgb_out=True):
    logger.debug(" - saving %s...", file_name)
    #copy
    imgout = np.array(arr)
    #clip
    imgout = np.clip(imgout, 0.0, 1.0)
    #rgb->srgb
    if srgb_out:
        rgbtosrgb(imgout)
    #unflattening
    imgout = imgout.reshape((resolution[1], resolution[0], 3))
    plt.imsave(file_name, imgout)
