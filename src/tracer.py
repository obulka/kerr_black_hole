#!/usr/bin/env python3
# Standard Imports
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
from black_hole_tracer.image_utils import *
from black_hole_tracer.output_utils import *
from black_hole_tracer.vec_utils import *


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
)
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_DIR)
TEXTURE_DIR = "textures"
TEXTURE_FILE = "grid.png"

DISABLE_SHUFFLING = False

NTHREADS = 16
CHUNKSIZE = 9000

default_options = {
    "resolution": "1512,762",
    "iterations": "5000", # Increase this for good results
    "camera_position": "2.6,1.570796,0.",
    "field_of_view": 1.5,
    "sRGB_in": "0",
    "sRGB_out": "0",
    "distort": "1", # 0: None
    "gain": "1",
    "normalize": "-1", # -1 for off
    "redshift": "1",
    "spin": ".998"
}
options = default_options

try:
    RESOLUTION = [int(x) for x in options["resolution"].split(",")]
    NITER = int(options["iterations"])

    CAMERA_POS = np.array([
        float(x) for x in options["camera_position"].split(",")
    ])

    #perform linear rgb->srgb conversion
    SRGBIN = int(options["sRGB_in"])
    SRGBOUT = int(options["sRGB_out"])

    DISTORT = int(options["distort"])
    GAIN = float(options["gain"])
    NORMALIZE = float(options["normalize"])
    REDSHIFT = float(options["redshift"])

    SPIN = float(options["spin"])
    SPIN_SQR = SPIN**2

except:
    logger.debug("error reading options: insufficient data")
    sys.exit()

logger.debug("%dx%d", RESOLUTION[0], RESOLUTION[1])


def show_progress(messtring, i, queue):
    mes = "Chunk %d/%d, %s" % (
        chunk_counters[i],
        len(schedules[i]),
        messtring.ljust(30)
    )
    queue.put((i, mes))


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


def rk4step(ode_fcn, t_0, delta_t, y_0, q):
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
    dy_0 = ode_fcn(y_0, t_0, q)

    y_1 = y_0 + dy_0 * delta_t / 2
    dy_1 = ode_fcn(y_1, t_0 + delta_t / 2, q)

    y_2 = y_0 + dy_1 * delta_t / 2
    dy_2 = ode_fcn(y_2, t_0 + delta_t / 2, q)

    y_3 = y_0 + dy_2 * delta_t
    dy_3 = ode_fcn(y_3, t_0 + delta_t, q)

    return y_0 + (dy_0 + 2 * dy_1 + 2 * dy_2 + dy_3) * delta_t / 6


def rk4_final(ode_fcn, t_span, y_0, q, horizon_radius):
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
        )
        prev_time = time

    return solution, mask


def ray_equation(rtp_vel, time, q):
    """ Solve the six ODEs """
    r, theta, phi, p_r, p_t, p_p = rtp_vel.T

    delta = _delta(r, SPIN_SQR)

    rho = _rho(r, theta, SPIN_SQR)
    rho_sqr = rho**2

    dr_dxi = delta * p_r / rho_sqr
    dtheta_dxi = p_t / rho_sqr

    delta_rho_sqr = delta * rho_sqr
    r_sqr = r**2

    cot_sqr = 1 / np.tan(theta)**2

    dphi_dxi = (
        (
            SPIN**3
            - SPIN_SQR * p_p
            - SPIN * delta
            + SPIN * r_sqr
            + p_p * delta * cot_sqr
            + p_p * delta
        ) / delta_rho_sqr
    )

    rho_fourth = rho_sqr**2

    P = SPIN_SQR - SPIN * p_p + r**2
    P_sqr = P**2
    two_del_rho_sqr = 2 * delta * rho_sqr
    P_term_2 = (p_p - SPIN)**2 + q
    R = P_sqr - delta * P_term_2
    cos_theta = np.cos(theta)

    dpr_dxi = (
        -p_r**2 * (SPIN_SQR * (r - 1) * cos_theta**2 + r * (r - SPIN_SQR))
        / rho_fourth
        + r * p_t**2 / rho_fourth
        + (4 * r * P - 2 * (r - 1) * P_term_2)
        / two_del_rho_sqr
        - (2 * (r - 1) * R)
        / (two_del_rho_sqr * delta)
        - (r * R) / (delta * rho_fourth)
        - r * (SPIN_SQR * cos_theta**2 - (p_p**2) * cot_sqr)
        / rho_fourth
    )

    sin_theta = np.sin(theta)
    Theta_term_2 = (p_p**2 / sin_theta**2 - SPIN_SQR) * cos_theta**2
    Theta = q - Theta_term_2

    dpt_dxi = (
        -SPIN_SQR * p_r**2 * (SPIN_SQR + (r - 2) * r) * sin_theta * cos_theta
        / rho_fourth
        - SPIN_SQR * p_t**2 * sin_theta * cos_theta
        / rho_fourth
        + SPIN_SQR * sin_theta * cos_theta * R
        / (delta * rho_fourth)
        + SPIN_SQR * sin_theta * cos_theta * Theta
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


def raytrace_schedule(i, schedule, total_shared, q):
    # this is the function running on each thread
    if len(schedule) == 0:
        return

    total_colour_buffer_preproc = to_numpy_array(total_shared, num_pixels)

    #schedule = schedules[i]

    iter_counters[i] = 0
    chunk_counters[i] = 0

    horizon_radius = 1.5 * (1 + np.sqrt(1 - SPIN_SQR))

    for chunk in schedule:
        chunk_counters[i] += 1

        #number of chunk pixels
        num_chunk_pixels = chunk.shape[0]

        #useful constant arrays 
        ones = np.ones((num_chunk_pixels))
        ones3 = np.ones((num_chunk_pixels, 3))
        UPFIELD = np.outer(ones, np.array([0., 1., 0.]))
        BLACK = np.outer(ones, np.array([0., 0., 0.]))

        # arrays of integer pixel coordinates
        x = chunk % RESOLUTION[0]
        y = chunk / RESOLUTION[0]

        show_progress("Generating view vectors...", i, q)

        #the view vector in 3D space
        view = np.ones((num_chunk_pixels, 2))

        view[:, 0] = np.pi * y.astype(float) / RESOLUTION[1]
        view[:, 1] = 2 * np.pi * x.astype(float) / RESOLUTION[0]

        r_c, theta_c, phi_c = CAMERA_POS
        delta = _delta(r_c, SPIN_SQR)
        sigma = _sigma(r_c, theta_c, SPIN_SQR, delta)
        rho = _rho(r_c, theta_c, SPIN_SQR)
        varpi = _varpi(theta_c, rho, sigma)
        alpha = _alpha(rho, delta, sigma)
        omega = _omega(r_c, SPIN, sigma)

        camera_vel_dir, camera_speed = equatorial_geodesic_orbit_velocity(
            r_c,
            SPIN,
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
            SPIN_SQR,
            momenta[..., 1],
            axial_angular_momentum,
        )
        # hit_horizon = ray_from_horizon_mask(
        #     r_c,
        #     SPIN,
        #     SPIN_SQR,
        #     carter_constant,
        #     axial_angular_momentum,
        #     momenta[..., 0],
        #     delta,
        # )
        # from_celestial_sphere = np.logical_not(hit_horizon)

        # Runge-Kutta
        bl_point = np.outer(ones, CAMERA_POS)
        rtp_vel = np.ones((num_chunk_pixels, 6))
        rtp_vel[:, 0:3] = bl_point
        rtp_vel[:, 3:6] = momenta

        times = np.linspace(0, -350, NITER)

        ode_fcn = ray_equation
        solutions, hit_horizon = rk4_final(ode_fcn, times, rtp_vel, carter_constant, horizon_radius)

        show_progress("generating sky layer...", i, q)

        vtheta = solutions[..., 1]
        vphi = solutions[..., 2]

        vuv = np.zeros((num_chunk_pixels, 2))

        vuv[:, 0] = np.mod(vphi, 2 * np.pi) / (2 * np.pi)
        vuv[:, 1] = np.mod(vtheta, np.pi) / (np.pi)

        col_bg = lookup(texarr_sky, vuv)[:, 0:3]

        show_progress("generating debug layers...", i, q)

        # hit_horizon = solutions[..., 0] < horizon_radius
        # col_bg[test] = np.ones_like(col_bg)[test]
        col_bg[hit_horizon] = np.zeros_like(col_bg)[hit_horizon]

        show_progress("blending layers...", i, q)

        col_bg_and_obj = col_bg

        show_progress("beaming back to mothership.", i, q)

        if not DISABLE_SHUFFLING:
            total_colour_buffer_preproc[chunk] = col_bg_and_obj
        else:
            total_colour_buffer_preproc[chunk[0]:(chunk[-1] + 1)] = col_bg_and_obj

        show_progress("garbage collection...", i, q)
        gc.collect()

    show_progress("Done.", i, q)


#ensuring existence of tests directory
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

logger.debug("Loading textures...")
texarr_sky = spm.imread(os.path.join(BASE_DIR, TEXTURE_DIR, TEXTURE_FILE))

# must convert to float here so we can work in linear colour
texarr_sky = texarr_sky.astype(float)
texarr_sky /= 255.0
if SRGBIN:
    # must do this before resizing to get correct results
    srgbtorgb(texarr_sky)

# maybe doing this manually and then loading is better.
logger.debug("(zooming sky texture...)")
texarr_sky = spm.imresize(texarr_sky, 2.0, interp="bicubic")
# imresize converts back to uint8 for whatever reason
texarr_sky = texarr_sky.astype(float)
texarr_sky /= 255.0


logger.debug("Computing rotation matrix...")

#array [0,1,2,...,num_pixels]
pixel_indices = np.arange(0, RESOLUTION[0] * RESOLUTION[1], 1)

#total number of pixels
num_pixels = pixel_indices.shape[0]

logger.debug("Generated %d pixel flattened array.", num_pixels)

# useful constant arrays
ones = np.ones((num_pixels))
ones3 = np.ones((num_pixels, 3))
UPFIELD = np.outer(ones,np.array([0., 1., 0.]))

# random sample of floats
random_sample = np.random.random_sample((num_pixels))


# PARTITIONING

# partition viewport in contiguous chunks
# CHUNKSIZE = 9000
if not DISABLE_SHUFFLING:
    np.random.shuffle(pixel_indices)

chunks = np.array_split(pixel_indices, num_pixels / CHUNKSIZE + 1)

NCHUNKS = len(chunks)

logger.debug("Split into %d chunks of %d pixels each", NCHUNKS, chunks[0].shape[0])

total_colour_buffer_preproc_shared = multi.Array(ctypes.c_float, num_pixels * 3)
total_colour_buffer_preproc = to_numpy_array(
    total_colour_buffer_preproc_shared,
    num_pixels,
)

# shuffle chunk list (does very good for equalizing load)
random.shuffle(chunks)

# partition chunk list in schedules for single threads
schedules = []

# from http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
q, r = divmod(NCHUNKS, NTHREADS)
indices = [q * i + min(i, r) for i in range(NTHREADS + 1)]

for i in range(NTHREADS):
    schedules.append(chunks[indices[i]:indices[i + 1]]) 

logger.debug(
    "Split list into %d schedules with %s chunks each",
    NTHREADS,
    ", ".join([str(len(schedule)) for schedule in schedules]),
)

# global clock start
start_time = time.time()

iter_counters = [0 for i in range(NTHREADS)]
chunk_counters = [0 for i in range(NTHREADS)]

#killers
killers = [False for i in range(NTHREADS)]

output = Outputter(NTHREADS)

# Threading
process_list = []
for i in range(NTHREADS):
    p = multi.Process(
        target=raytrace_schedule,
        args=(
            i,
            schedules[i],
            total_colour_buffer_preproc_shared,
            output.queue,
        )
    )
    process_list.append(p)

logger.debug("Starting threads...")

for proc in process_list:
    proc.start()

try:
    refreshcounter = 0
    while True:
        refreshcounter += 1
        time.sleep(0.1)
    
        output.parsemessages()

        output.setmessage("Idle.", -1)

        all_done = True

        for i in range(NTHREADS):
            if process_list[i].is_alive():
                all_done = False

        if all_done:
            break

except KeyboardInterrupt:
    for i in range(NTHREADS):
        killers[i] = True
    sys.exit()

del output

logger.debug("Done tracing.")

logger.debug(
    "Total raytracing time: %s",
    datetime.timedelta(seconds=(time.time() - start_time)),
)

logger.debug("Postprocessing...")

#gain
logger.debug("- gain...")
total_colour_buffer_preproc *= GAIN

colour = total_colour_buffer_preproc

#normalization
if NORMALIZE > 0:
    logger.debug("- normalizing...")
    colour *= 1 / (NORMALIZE * np.amax(colour.flatten()))

#final colour
colour = np.clip(colour, 0., 1.)

logger.debug("Conversion to image and saving...")

save_to_img(
    colour,
    os.path.join(OUTPUT_PATH, "out.png"),
    RESOLUTION,
    srgb_out=SRGBOUT,
)
save_to_img(
    total_colour_buffer_preproc,
    os.path.join(OUTPUT_PATH, "out.png"),
    RESOLUTION,
    srgb_out=SRGBOUT,
)
