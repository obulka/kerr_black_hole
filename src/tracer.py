#!/usr/bin/env python3
""" Raytrace a Kerr Black Hole

Author: Owen Bulka
"""
# Standard Imports
import argparse
import datetime
import multiprocessing as multi
import os
import time
import sys

# 3rd Party Imports
import numpy as np
import scipy.misc as spm

# Local Imports
from black_hole_tracer.kerr import KerrBlackHole, KerrRaytracer
from black_hole_tracer.utils.image import (
    convert_image_to_float,
    post_process,
    save_to_img,
    srgbtorgb,
)
from black_hole_tracer.utils.output import print_help_and_exit


def parse_args(defaults):
    """ Parse the arguments.

    Args:
        defaults (dict): The default parameters.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Ray trace a spinning black hole.")
    parser.add_argument(
        "-a",
        default=defaults["spin"],
        dest="spin",
        help=(
            "spin parameter a/M where M is 1. "
            "Default: {}"
        ).format(defaults["spin"]),
        type=float,
    )
    parser.add_argument(
        "-t",
        default=defaults["texture_path"],
        dest="texture_path",
        help=(
            "Path to the texture to apply the warping of spacetime to. "
            "Default: {}"
        ).format(defaults["texture_path"]),
        type=str,
    )
    parser.add_argument(
        "-o",
        default=defaults["output_path"],
        dest="output_path",
        help=(
            "Path to save the final render to (include filename). "
            "Default: {}"
        ).format(defaults["output_path"]),
        type=str,
    )
    parser.add_argument(
        "-r",
        default=",".join(str(value) for value in defaults["resolution"]),
        dest="resolution",
        help=(
            "Comma separated render resolution. "
            "Default: {}"
        ).format(defaults["resolution"]),
        type=str,
    )
    parser.add_argument(
        "-i",
        default=defaults["iterations"],
        dest="iterations",
        help=(
            "Number of steps to take during integration. "
            "Higher values result in higher quality renders. "
            "Default: {}"
        ).format(defaults["iterations"]),
        type=int,
    )
    parser.add_argument(
        "-p",
        default=",".join(str(value) for value in defaults["camera_position"]),
        dest="camera_position",
        help=(
            "Comma separated position of the camera in spherical coordinates. "
            "The corridinate order is radial, theta, phi. "
            "Default: {}"
        ).format(defaults["camera_position"]),
        type=str,
    )
    parser.add_argument(
        "-j",
        default=defaults["num_processes"],
        dest="num_processes",
        help=(
            "Number of processes to render in. "
            "Default: {}"
        ).format(defaults["num_processes"]),
        type=int,
    )
    parser.add_argument(
        "-c",
        default=defaults["chunk_size"],
        dest="chunk_size",
        help=(
            "Number of pixels per chunk. "
            "Default: {}"
        ).format(defaults["chunk_size"]),
        type=int,
    )
    parser.add_argument(
        "-g",
        default=defaults["gain"],
        dest="gain",
        help=(
            "Gain to apply during postprocessing. "
            "Default: {}"
        ).format(defaults["gain"]),
        type=int,
    )
    parser.add_argument(
        "-n",
        default=defaults["normalize"],
        dest="normalize",
        help=(
            "Normalization factor to apply during postprocessing. "
            "Default: {}"
        ).format(defaults["normalize"]),
        type=int,
    )
    parser.add_argument(
        "--no_srgb",
        action="store_true",
        dest="no_srgb",
        help="Do not convert input and output to sRGB. ",
    )
    parser.add_argument(
        "--disable_shuffle",
        action="store_true",
        dest="disable_shuffle",
        help="Disable the shuffling of pixels before dividing into chunks.",
    )
    args = parser.parse_args()

    try:
        args.resolution = tuple(
            int(pixels.strip()) for pixels in args.resolution.split(",")
        )
        args.camera_position = np.array([
            float(coord.strip()) for coord in args.camera_position.split(",")
        ])

    except ValueError as error:
        print(error)
        print("\nUse the -h switch to see usage information.\n")
        sys.exit()
    return args


def main():
    """ Render a spinning black hole. """
    base_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
    )
    default_output_path = os.path.join(base_dir, "output", "out.png")
    default_texture_path = os.path.join(base_dir, "textures", "grid.png")

    default_options = {
        "resolution": (1512, 762),
        "texture_path": default_texture_path,
        "output_path": default_output_path,
        "iterations": 200, # Increase this for good results
        "camera_position": [2.6, 1.570796, 0.],
        "num_processes": multi.cpu_count(),
        "chunk_size": 9000,
        "gain": 1,
        "normalize": 0,
        "spin": 0.998,
    }
    args = parse_args(default_options)

    output_path = os.path.dirname(args.output_path)
    if not os.path.exists(output_path):
        print("Error: Output path does not exist at:")
        print(args.output_path)
        print("Create the directory or change the path then try again.")
        print_help_and_exit()


    try:
        texture = spm.imread(args.texture_path)
    except FileNotFoundError as error:
        print(error)
        print("Error: Texture file not found at:")
        print(args.texture_path)
        print_help_and_exit()

    # Convert to float to work in linear colour space
    texture = convert_image_to_float(texture)
    if not args.no_srgb:
        # Convert to sRGB before resizing for correct results
        srgbtorgb(texture)

    texture = convert_image_to_float(
        spm.imresize(texture, 2.0, interp="bicubic"),
    )

    black_hole = KerrBlackHole(args.spin)
    raytracer = KerrRaytracer(
        black_hole,
        args.camera_position,
        texture,
        args.resolution,
        args.iterations,
        args.num_processes,
        args.chunk_size,
        shuffle=not args.disable_shuffle,
    )
    raytracer.generate_image()
    print("Raytracing Completed Succesfully.")
    print(
        "Total raytracing time:",
        datetime.timedelta(seconds=(time.time() - raytracer.start_time)),
    )

    colour = post_process(raytracer.colour_buffer_preproc, args.gain, args.normalize)

    save_to_img(
        colour,
        args.output_path,
        args.resolution,
        srgb_out=not args.no_srgb,
    )


if __name__ == "__main__":
    main()
