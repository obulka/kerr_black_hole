#!/usr/bin/env python3
# Standard Imports
import argparse
import ctypes
import datetime
import gc
import logging
import multiprocessing as multi
import os, random, sys, time

# 3rd Party Imports
import numpy as np
import scipy.misc as spm
import scipy.ndimage as ndim

# Local Imports
from black_hole_tracer.utils.image import *
from black_hole_tracer.utils.output import *
from black_hole_tracer.utils.vector import *


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def parse_args(defaults):
    """Parse the arguments.

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
        default=defaults["resolution"],
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
        default=defaults["camera_position"],
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
        "--srgb_in",
        default=defaults["srgb_in"],
        dest="srgb_in",
        help=(
            "Convert input texture to sRGB. "
            "Default: {}"
        ).format(defaults["srgb_in"]),
        type=bool,
    )
    parser.add_argument(
        "--srgb_out",
        default=defaults["srgb_out"],
        dest="srgb_out",
        help=(
            "Save output render as sRGB. "
            "Default: {}"
        ).format(defaults["srgb_out"]),
        type=bool,
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
        "--disable_shuffle",
        action="store_true",
        dest="disable_shuffle",
        help="Disable the shuffling of pixels before dividing into chunks.",
    )
    args = parser.parse_args()

    try:
        args.resolution = [int(pixels.strip()) for pixels in args.resolution.split(",")]
        args.camera_position = [
            float(coord.strip()) for coord in args.camera_position.split(",")
        ]

    except ValueError as error:
        print(error)
        print("\nUse the -h switch to see usage information.\n")
        exit()
    return args


def equatorial_geodesic_orbit_velocity(r_c, a, omega, varpi, alpha):
    return np.array([0, 0, 1.]), varpi * ((1 / (a + r_c**1.5)) - omega) / alpha


def _delta(r, a_sqr):
    return r**2 - 2 * r + a_sqr


def _rho(r, theta, a_sqr):
    return np.sqrt(r**2 + a_sqr * np.cos(theta)**2)


def _sigma(r, theta, a_sqr, delta):
    return np.sqrt(
        (r**2 + a_sqr)**2
        - a_sqr * delta * np.sin(theta)**2
    )


def _alpha(rho, delta, sigma):
    return rho * np.sqrt(delta) / sigma


def _omega(r, a, sigma):
    return 2 * r * a / sigma**2


def _varpi(theta, rho, sigma):
    return sigma * np.sin(theta) / rho


def _Theta(theta, q, b, a_sqr):
    return q - ((b / np.sin(theta))**2 - a_sqr) * np.cos(theta)**2


def relativistic_aberration(cart_ray_dir, camera_vel_dir, camera_speed):
    """Note speed of light is 1"""
    B_r, B_theta, B_phi = camera_vel_dir
    N_x, N_y, N_z = cart_ray_dir.T
    denom = 1 - camera_speed * N_y
    rel_speed = -np.sqrt(1 - camera_speed**2)

    n_Fx = rel_speed * N_x / denom
    n_Fy = (camera_speed - N_y) / denom
    n_Fz = rel_speed * N_z / denom

    kappa = np.sqrt(1 - B_theta**2)

    n_Fr = B_phi * n_Fx / kappa + B_r * n_Fy + B_r * B_theta * n_Fz / kappa
    n_Ftheta = B_theta * n_Fy - kappa * n_Fz
    n_Fphi = -B_r * n_Fx / kappa + B_phi * n_Fy + B_theta * B_phi * n_Fz / kappa

    return np.array([n_Fr, n_Ftheta, n_Fphi]).T


def canonical_momenta(rel_aberration, rho, delta, alpha, omega, varpi):
    E_F = 1 / (alpha + omega * varpi * rel_aberration[..., 2])
    return np.multiply(E_F[:, np.newaxis], rel_aberration) * np.array([
        rho / np.sqrt(delta),
        rho,
        varpi,
    ])


def compute_carter_constant(theta, a_sqr, axial_momentum, azimuthal_momentum):
    return (
        azimuthal_momentum**2
        + ((axial_momentum / np.sin(theta))**2 - a_sqr)
        * np.cos(theta)**2
    )


def largest_real_root_R(b, a, a_sqr, q):
    coeffs = np.zeros((len(b), 5))
    coeffs[..., 0] = 1.
    coeffs[..., 2] = a_sqr - b**2 - q
    coeffs[..., 3] = 2 * (a_sqr - 2 * a * b + b**2 + q)
    coeffs[..., 4] = -q * a**2

    real_max_roots = []
    for row in coeffs:
        roots = np.roots(row)
        real_max_roots.append(roots[np.isreal(roots)].max().real)

    return np.array(real_max_roots)


def ray_from_horizon_mask(
        r,
        a,
        a_sqr,
        carter_constant,
        axial_angular_momentum,
        radial_momentum,
        delta):
    def b_(r_, a, a_sqr):
        return -(r_**3 - 3 * r_**2 + a_sqr * r_ + a_sqr) / (a * (r_ - 1))

    def q_(r_, a_sqr):
        return (
            -((r_**3) * (r_**3 - 6 * r_**2 + 9 * r_ - 4 * a_sqr))
            / (a_sqr * (r_ - 1)**2)
        )

    r_12 = 2 * (1 + np.cos(2 * np.arccos(np.array([-a, a])) / 3))

    b_21 = b_(r_12, a, a_sqr)

    b_condition = np.logical_and(
        b_21[1] < axial_angular_momentum,
        axial_angular_momentum < b_21[0],
    )

    q_0 = q_(axial_angular_momentum, a_sqr)

    q_condition = carter_constant < q_0

    horizon_condition = np.logical_and(b_condition, q_condition)

    # # Two secondary conditions
    if horizon_condition.any():
        positive_radial_momentum = radial_momentum > 0

    if horizon_condition.all():
        return positive_radial_momentum
    else:
        r_up = largest_real_root_R(
            axial_angular_momentum,
            a,
            a_sqr,
            carter_constant,
        )
        r_cam_less_r_up = r < r_up
        return r_cam_less_r_up

        if not horizon_condition.any():
            return r_cam_less_r_up

    r_cam_less_r_up[horizon_condition] = positive_radial_momentum[horizon_condition]

    return r_cam_less_r_up


def rk4step(ode_fcn, t_0, delta_t, y_0, q, spin, spin_sqr):
    """ Compute one rk4 step.

    Args:
        ode_fcn (function):
            Function handle for right hand sides of ODEs
            (Returns: np.array(N, float)).

        initial_time (float): Initial value of independent variable.

        time_step (float): Time step.

        y_0 (np.array(N, float)): Initial values.

    Returns:
        np.array(N, float): Final values.
    """
    dy_0 = ode_fcn(y_0, t_0, q, spin, spin_sqr)

    y_1 = y_0 + dy_0 * delta_t / 2
    dy_1 = ode_fcn(y_1, t_0 + delta_t / 2, q, spin, spin_sqr)

    y_2 = y_0 + dy_1 * delta_t / 2
    dy_2 = ode_fcn(y_2, t_0 + delta_t / 2, q, spin, spin_sqr)

    y_3 = y_0 + dy_2 * delta_t
    dy_3 = ode_fcn(y_3, t_0 + delta_t, q, spin, spin_sqr)

    return y_0 + (dy_0 + 2 * dy_1 + 2 * dy_2 + dy_3) * delta_t / 6


def rk4_final(ode_fcn, t_span, y_0, q, horizon_radius, spin, spin_sqr):
    """ N_out is the length of the `t_span` parameter.
        N is the length of the `y_0` parameter.
        Only returns the final solution to save memory.

    Args:
        ode_fcn (function):
            Function handle for right hand sides of ODEs
            (Returns: np.array(N, float)).

        t_span (np.array(N_out, float)): Vector of output times.

        y_0 (np.array(N, float)): Initial values.

    Returns:
        np.array(N, float): Output values.
    """
    # This loop structure accommodates negative time steps
    prev_time = t_span[0]
    solution = y_0.copy()
    mask = np.zeros(solution.shape[0])

    for time in t_span[1:]:
        mask = np.logical_or(mask, solution[..., 0] < horizon_radius)
        solution = rk4step(
            ode_fcn,
            prev_time,
            time - prev_time,
            solution,
            q,
            spin,
            spin_sqr,
        )
        prev_time = time

    return solution, mask


def ray_equation(rtp_vel, time, q, spin, spin_sqr):
    """ Solve the six ODEs """
    r, theta, phi, p_r, p_t, p_p = rtp_vel.T

    delta = _delta(r, spin_sqr)

    rho = _rho(r, theta, spin_sqr)
    rho_sqr = rho**2

    dr_dxi = delta * p_r / rho_sqr
    dtheta_dxi = p_t / rho_sqr

    delta_rho_sqr = delta * rho_sqr
    r_sqr = r**2

    cot_sqr = 1 / np.tan(theta)**2

    dphi_dxi = (
        (
            spin**3
            - spin_sqr * p_p
            - spin * delta
            + spin * r_sqr
            + p_p * delta * cot_sqr
            + p_p * delta
        ) / delta_rho_sqr
    )

    rho_fourth = rho_sqr**2

    P = spin_sqr - spin * p_p + r**2
    P_sqr = P**2
    two_del_rho_sqr = 2 * delta * rho_sqr
    P_term_2 = (p_p - spin)**2 + q
    R = P_sqr - delta * P_term_2
    cos_theta = np.cos(theta)

    dpr_dxi = (
        -p_r**2 * (spin_sqr * (r - 1) * cos_theta**2 + r * (r - spin_sqr))
        / rho_fourth
        + r * p_t**2 / rho_fourth
        + (4 * r * P - 2 * (r - 1) * P_term_2)
        / two_del_rho_sqr
        - (2 * (r - 1) * R)
        / (two_del_rho_sqr * delta)
        - (r * R) / (delta * rho_fourth)
        - r * (spin_sqr * cos_theta**2 - (p_p**2) * cot_sqr)
        / rho_fourth
    )

    sin_theta = np.sin(theta)
    Theta_term_2 = (p_p**2 / sin_theta**2 - spin_sqr) * cos_theta**2
    Theta = q - Theta_term_2

    dpt_dxi = (
        -spin_sqr * p_r**2 * (spin_sqr + (r - 2) * r) * sin_theta * cos_theta
        / rho_fourth
        - spin_sqr * p_t**2 * sin_theta * cos_theta
        / rho_fourth
        + spin_sqr * sin_theta * cos_theta * R
        / (delta * rho_fourth)
        + spin_sqr * sin_theta * cos_theta * Theta
        / rho_fourth
        + (sin_theta * Theta_term_2 + p_p**2 / np.tan(theta)**3)
        / rho_sqr 
    )
    dpp_dxi = np.zeros_like(dpt_dxi)

    return np.array([
        dr_dxi,
        dtheta_dxi,
        dphi_dxi,
        dpr_dxi,
        dpt_dxi,
        dpp_dxi,
    ]).T


# Disgusting arguments, initialize in a class
def scheduled_raytrace(
        i,
        schedule,
        total_shared,
        q,
        show_progress,
        num_pixels,
        iter_counters,
        chunk_counters,
        spin,
        resolution,
        camera_position,
        iterations,
        disable_shuffle,
        texture):
    # this is the function running on each thread
    if len(schedule) == 0:
        return

    total_colour_buffer_preproc = to_numpy_array(total_shared, num_pixels)

    iter_counters[i] = 0 # Increment in rk4 if we want to add time back
    chunk_counters[i] = 0

    spin_sqr = spin**2
    horizon_radius = 1.5 * (1 + np.sqrt(1 - spin_sqr))

    for chunk in schedule:
        chunk_counters[i] += 1

        #number of chunk pixels
        num_chunk_pixels = chunk.shape[0]

        #useful constant arrays 
        ones = np.ones((num_chunk_pixels))
        BLACK = np.outer(ones, np.array([0., 0., 0.]))

        # arrays of integer pixel coordinates
        x = chunk % resolution[0]
        y = chunk / resolution[0]

        show_progress("Generating view vectors...", i, q)

        #the view vector in 3D space
        view = np.ones((num_chunk_pixels, 2))

        view[:, 0] = np.pi * y.astype(float) / resolution[1]
        view[:, 1] = 2 * np.pi * x.astype(float) / resolution[0]

        r_c, theta_c, phi_c = camera_position
        delta = _delta(r_c, spin_sqr)
        sigma = _sigma(r_c, theta_c, spin_sqr, delta)
        rho = _rho(r_c, theta_c, spin_sqr)
        varpi = _varpi(theta_c, rho, sigma)
        alpha = _alpha(rho, delta, sigma)
        omega = _omega(r_c, spin, sigma)

        camera_vel_dir, camera_speed = equatorial_geodesic_orbit_velocity(
            r_c,
            spin,
            omega,
            varpi,
            alpha,
        )

        theta_cs, phi_cs = view.T
        norm_view = np.array([
            np.sin(theta_cs) * np.cos(phi_cs),
            np.sin(theta_cs) * np.sin(phi_cs),
            np.cos(theta_cs),
        ]).T

        rel_aberration = relativistic_aberration(
            norm_view,
            camera_vel_dir,
            camera_speed,
        )
        momenta = canonical_momenta(rel_aberration, rho, delta, alpha, omega, varpi)
        axial_angular_momentum = momenta[..., 2]
        carter_constant = compute_carter_constant(
            theta_c,
            spin_sqr,
            momenta[..., 1],
            axial_angular_momentum,
        )
        # hit_horizon = ray_from_horizon_mask(
        #     r_c,
        #     spin,
        #     spin_sqr,
        #     carter_constant,
        #     axial_angular_momentum,
        #     momenta[..., 0],
        #     delta,
        # )
        # from_celestial_sphere = np.logical_not(hit_horizon)

        # Runge-Kutta
        bl_point = np.outer(ones, camera_position)
        rtp_vel = np.ones((num_chunk_pixels, 6))
        rtp_vel[:, 0:3] = bl_point
        rtp_vel[:, 3:6] = momenta

        times = np.linspace(0, -350, iterations)

        ode_fcn = ray_equation
        solutions, hit_horizon = rk4_final(
            ode_fcn,
            times,
            rtp_vel,
            carter_constant,
            horizon_radius,
            spin,
            spin_sqr,
        )

        show_progress("generating sky layer...", i, q)

        vtheta = solutions[..., 1]
        vphi = solutions[..., 2]

        vuv = np.zeros((num_chunk_pixels, 2))

        vuv[:, 0] = np.mod(vphi, 2 * np.pi) / (2 * np.pi)
        vuv[:, 1] = np.mod(vtheta, np.pi) / (np.pi)

        col_bg = lookup(texture, vuv)[:, 0:3]

        show_progress("generating debug layers...", i, q)

        # hit_horizon = solutions[..., 0] < horizon_radius
        # col_bg[test] = np.ones_like(col_bg)[test]
        col_bg[hit_horizon] = np.zeros_like(col_bg)[hit_horizon]

        show_progress("blending layers...", i, q)

        col_bg_and_obj = col_bg

        show_progress("beaming back to mothership.", i, q)

        if not disable_shuffle:
            total_colour_buffer_preproc[chunk] = col_bg_and_obj
        else:
            total_colour_buffer_preproc[chunk[0]:(chunk[-1] + 1)] = col_bg_and_obj

        show_progress("garbage collection...", i, q)
        gc.collect()

    show_progress("Done.", i, q)


def main():
    """ Render a spinning black hole. """
    base_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
    )
    default_output_path = os.path.join(base_dir, "output", "out.png")
    default_texture_path = os.path.join(base_dir, "textures", "grid.png")

    default_options = {
        "resolution": "1512,762",
        "texture_path": default_texture_path,
        "output_path": default_output_path,
        "iterations": 500, # Increase this for good results
        "camera_position": "2.6,1.570796,0.",
        "num_processes": multi.cpu_count(),
        "chunk_size": 9000,
        "srgb_in": True,
        "srgb_out": True,
        "gain": 1,
        "normalize": 0,
        "spin": 0.998,
    }
    args = parse_args(default_options)

    # Ensure output path exists
    LOGGER.debug("Checking output location...")

    output_path = os.path.dirname(args.output_path)
    if not os.path.exists(output_path):
        print("Error: Output path does not exist at:")
        print(args.output_path)
        print("Create the directory or change the path then try again.")
        print_help_and_exit()

    LOGGER.debug("Loading texture...")

    try:
        texture = spm.imread(args.texture_path)
    except FileNotFoundError as error:
        print("Error: Texture file not found at:")
        print(args.texture_path)
        print_help_and_exit()


    # Convert to float to work in linear colour space
    texture = convert_image_to_float(texture)
    if args.srgb_in:
        # Convert to sRGB before resizing for correct results
        srgbtorgb(texture)

    LOGGER.debug("Scaling input texture...")
    texture = convert_image_to_float(
        spm.imresize(texture, 2.0, interp="bicubic"),
    )

    LOGGER.debug("Preparing for multiprocessing...")

    num_pixels = args.resolution[0] * args.resolution[1]
    pixel_indices = np.arange(0, num_pixels)

    LOGGER.debug("Splitting into chunks...")

    if not args.disable_shuffle:
        np.random.shuffle(pixel_indices)

    chunks = np.array_split(pixel_indices, num_pixels / args.chunk_size + 1)

    LOGGER.debug(
        "Split into %d chunks of %d pixels each",
        len(chunks),
        chunks[0].shape[0],
    )

    total_colour_buffer_preproc_shared = multi.Array(ctypes.c_float, num_pixels * 3)
    total_colour_buffer_preproc = to_numpy_array(
        total_colour_buffer_preproc_shared,
        num_pixels,
    )

    # Shuffle chunks to equalize load
    random.shuffle(chunks)

    # partition chunk list in schedules for single threads
    schedules = []

    # from http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
    ########## Change this after all else tested
    q, r = divmod(len(chunks), args.num_processes)
    indices = [q * i + min(i, r) for i in range(args.num_processes + 1)]

    for i in range(args.num_processes):
        schedules.append(chunks[indices[i]:indices[i + 1]])
    ##########

    LOGGER.debug(
        "Split load into %d processes with %s chunks in each, respectively...",
        args.num_processes,
        ", ".join([str(len(schedule)) for schedule in schedules]),
    )

    # Start clock to time raytracing
    start_time = time.time()

    iter_counters = np.zeros(args.num_processes).astype(int) # Add to rk4 or delete
    chunk_counters = np.zeros(args.num_processes).astype(int)
    killers = np.zeros(args.num_processes).astype(bool)

    output = Outputter(args.num_processes)

    show_progress = init_show_progress(chunk_counters, schedules)

    # Multiprocessing
    process_list = []
    for process_num in range(args.num_processes):
        process = multi.Process(
            target=scheduled_raytrace,
            args=(
                process_num,
                schedules[process_num],
                total_colour_buffer_preproc_shared,
                output.queue,
                show_progress,
                num_pixels,
                iter_counters,
                chunk_counters,
                args.spin,
                args.resolution,
                args.camera_position,
                args.iterations,
                args.disable_shuffle,
                texture,
            )
        )
        process_list.append(process)

    LOGGER.debug("Starting processes...")

    ##### Try moving this ^^
    for process in process_list:
        process.start()

    try:
        refreshcounter = 0
        while True:
            refreshcounter += 1
            time.sleep(0.1)
        
            output.parsemessages()

            output.setmessage("Idle.", -1)

            all_done = True
            for process in process_list:
                if process.is_alive():
                    all_done = False
                    break

            if all_done:
                break

    except KeyboardInterrupt:
        for process_num in range(args.num_processes):
            killers[process_num] = True
        sys.exit()

    del output

    LOGGER.debug("Finished raytracing...")

    LOGGER.debug(
        "Total raytracing time: %s",
        datetime.timedelta(seconds=(time.time() - start_time)),
    )

    LOGGER.debug("Postprocessing...")

    # Gain
    LOGGER.debug("Applying gain...")
    total_colour_buffer_preproc *= args.gain

    colour = total_colour_buffer_preproc

    # Normalization
    if args.normalize > 0:
        LOGGER.debug("Applying normalization...")
        max_pixel = np.amax(colour.flatten())
        if max_pixel:
            colour *= 1 / (args.normalize * max_pixel)

    # Final image colour
    colour = np.clip(colour, 0., 1.)

    LOGGER.debug("Conversion to final image and saving...")

    save_to_img(
        colour,
        args.output_path,
        args.resolution,
        srgb_out=args.srgb_out,
    )


if __name__ == "__main__":
    main()
