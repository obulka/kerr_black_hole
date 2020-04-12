""" Kerr Black Hole """
# Standard Imports
import gc

# 3rd Party Imports
import numpy as np

# Local Imports
from .image_generation import ScheduledImageGenerator
from .utils.image import lookup


class KerrBlackHole:
    """ Kerr Black Hole implementation """

    def __init__(self, spin):
        self._spin = spin
        self._spin_sqr = self._spin**2
        self._horizon_radius = 1.1 * (1 + np.sqrt(1 - self._spin_sqr))

    @property
    def spin(self):
        """"""
        return self._spin

    @property
    def spin_sqr(self):
        """"""
        return self._spin_sqr

    @property
    def horizon_radius(self):
        """"""
        return self._horizon_radius

    @staticmethod
    def equatorial_geodesic_orbit_velocity(r_c, a, omega, varpi, alpha):
        """
        """
        return np.array([0, 0, 1.]), varpi * ((1 / (a + r_c**1.5)) - omega) / alpha

    @staticmethod
    def delta(r, a_sqr):
        """
        """
        return r**2 - 2 * r + a_sqr

    @staticmethod
    def rho(r, theta, a_sqr):
        """
        """
        return np.sqrt(r**2 + a_sqr * np.cos(theta)**2)

    @staticmethod
    def sigma(r, theta, a_sqr, delta):
        """
        """
        return np.sqrt(
            (r**2 + a_sqr)**2
            - a_sqr * delta * np.sin(theta)**2
        )

    @staticmethod
    def alpha(rho, delta, sigma):
        """
        """
        return rho * np.sqrt(delta) / sigma

    @staticmethod
    def omega(r, a, sigma):
        """
        """
        return 2 * r * a / sigma**2

    @staticmethod
    def varpi(theta, rho, sigma):
        """
        """
        return sigma * np.sin(theta) / rho

    @staticmethod
    def Theta(theta, q, b, a_sqr):
        """
        """
        return q - ((b / np.sin(theta))**2 - a_sqr) * np.cos(theta)**2

    @staticmethod
    def relativistic_aberration(cart_ray_dir, camera_vel_dir, camera_speed):
        """ Note speed of light is 1. """
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

    @staticmethod
    def canonical_momenta(rel_aberration, rho, delta, alpha, omega, varpi):
        """
        """
        E_F = 1 / (alpha + omega * varpi * rel_aberration[..., 2])
        return np.multiply(E_F[:, np.newaxis], rel_aberration) * np.array([
            rho / np.sqrt(delta),
            rho,
            varpi,
        ])

    @staticmethod
    def compute_carter_constant(theta, a_sqr, axial_momentum, azimuthal_momentum):
        """
        """
        return (
            azimuthal_momentum**2
            + ((axial_momentum / np.sin(theta))**2 - a_sqr)
            * np.cos(theta)**2
        )

    @staticmethod
    def largest_real_root_R(b, a, a_sqr, q):
        """
        """
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

    @staticmethod
    def ray_from_horizon_mask(
            r,
            a,
            a_sqr,
            carter_constant,
            axial_angular_momentum,
            radial_momentum,
            delta):
        """
        """
        def b_(r_, a, a_sqr):
            """
            """
            return -(r_**3 - 3 * r_**2 + a_sqr * r_ + a_sqr) / (a * (r_ - 1))

        def q_(r_, a_sqr):
            """
            """
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

        q_0 = q_(r, a_sqr)

        q_condition = carter_constant < q_0

        horizon_condition = np.logical_and(b_condition, q_condition)

        # # Two secondary conditions
        if horizon_condition.any():
            positive_radial_momentum = radial_momentum > 0

        if horizon_condition.all():
            return positive_radial_momentum
        else:
            r_up = KerrBlackHole.largest_real_root_R(
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

    @staticmethod
    def ray_equation(rtp_vel, q, spin, spin_sqr):
        """ Solve the six ODEs """
        r, theta, _, p_r, p_t, p_p = rtp_vel.T

        delta = KerrBlackHole.delta(r, spin_sqr)

        rho = KerrBlackHole.rho(r, theta, spin_sqr)
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


class KerrRaytracer(ScheduledImageGenerator):
    """ Raytrace a Kerr black hole. """

    def __init__(
            self,
            kerr_black_hole,
            camera_position,
            texture,
            resolution,
            iterations,
            num_processes,
            chunk_size,
            shuffle=True):
        super(KerrRaytracer, self).__init__(
            resolution,
            num_processes,
            chunk_size,
            shuffle=shuffle,
        )
        self._kerr_black_hole = kerr_black_hole
        self._camera_position = camera_position
        self._texture = texture
        self._iterations = iterations

    @staticmethod
    def _rk4_step(ode_fcn, delta_xi, y_0, q, spin, spin_sqr):
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
        dy_0 = ode_fcn(y_0, q, spin, spin_sqr)

        y_1 = y_0 + dy_0 * delta_xi / 2
        dy_1 = ode_fcn(y_1, q, spin, spin_sqr)

        y_2 = y_0 + dy_1 * delta_xi / 2
        dy_2 = ode_fcn(y_2, q, spin, spin_sqr)

        y_3 = y_0 + dy_2 * delta_xi
        dy_3 = ode_fcn(y_3, q, spin, spin_sqr)

        return y_0 + (dy_0 + 2 * dy_1 + 2 * dy_2 + dy_3) * delta_xi / 6

    @staticmethod
    def _rk4_final(ode_fcn, xi_span, y_0, q, horizon_radius, spin, spin_sqr):
        """ N_out is the length of the `xi_span` parameter.
            N is the length of the `y_0` parameter.
            Only returns the final solution to save memory.

        Args:
            ode_fcn (function):
                Function handle for right hand sides of ODEs
                (Returns: np.array(N, float)).

            xi_span (np.array(N_out, float)): Vector of output times.

            y_0 (np.array(N, float)): Initial values.

        Returns:
            np.array(N, float): Output values.
        """
        # This loop structure accommodates negative time steps
        prev_time = xi_span[0]
        solution = y_0.copy()
        mask = np.zeros(solution.shape[0])

        for time in xi_span[1:]:
            mask = np.logical_or(mask, solution[..., 0] < horizon_radius)
            solution = KerrRaytracer._rk4_step(
                ode_fcn,
                time - prev_time,
                solution,
                q,
                spin,
                spin_sqr,
            )
            prev_time = time

        return solution, mask

    def _scheduled_generation(self, process_num, schedule):
        """
        """
        if len(schedule) == 0:
            return

        self._chunk_counters[process_num] = 0

        for chunk in schedule:
            self._chunk_counters[process_num] += 1

            num_chunk_pixels = chunk.shape[0]

            ones = np.ones((num_chunk_pixels))

            x = chunk % self._resolution[0]
            y = chunk / self._resolution[0]

            self._show_progress("Generating view vectors...", process_num)

            view = np.ones((num_chunk_pixels, 2))

            view[:, 0] = np.pi * y.astype(float) / self._resolution[1]
            view[:, 1] = 2 * np.pi * x.astype(float) / self._resolution[0] - np.pi

            r_c, theta_c, _ = self._camera_position
            spin_sqr = self._kerr_black_hole.spin_sqr
            spin = self._kerr_black_hole.spin

            delta = self._kerr_black_hole.delta(r_c, spin_sqr)
            sigma = self._kerr_black_hole.sigma(r_c, theta_c, spin_sqr, delta)
            rho = self._kerr_black_hole.rho(r_c, theta_c, spin_sqr)
            varpi = self._kerr_black_hole.varpi(theta_c, rho, sigma)
            alpha = self._kerr_black_hole.alpha(rho, delta, sigma)
            omega = self._kerr_black_hole.omega(r_c, spin, sigma)

            camera_vel_dir, camera_speed = self._kerr_black_hole.equatorial_geodesic_orbit_velocity(
                r_c,
                spin,
                omega,
                varpi,
                alpha,
            )

            theta_cs, phi_cs = view.T
            norm_view = np.array([
                -np.sin(theta_cs) * np.cos(phi_cs),
                -np.sin(theta_cs) * np.sin(phi_cs),
                -np.cos(theta_cs),
            ]).T

            rel_aberration = self._kerr_black_hole.relativistic_aberration(
                norm_view,
                camera_vel_dir,
                camera_speed,
            )
            momenta = self._kerr_black_hole.canonical_momenta(
                rel_aberration,
                rho,
                delta,
                alpha,
                omega,
                varpi,
            )
            axial_angular_momentum = momenta[..., 2]
            carter_constant = self._kerr_black_hole.compute_carter_constant(
                theta_c,
                spin_sqr,
                momenta[..., 1],
                axial_angular_momentum,
            )
            ## This was the attempt at the:
            ## "Gravitational lensing by spinning black holes in astrophysics,
            ## and in the movie Interstellar" Implementation that has yet to work
            # hit_horizon = KerrBlackHole.ray_from_horizon_mask(
            #     r_c,
            #     spin,
            #     spin_sqr,
            #     carter_constant,
            #     axial_angular_momentum,
            #     momenta[..., 0],
            #     delta,
            # )

            bl_point = np.outer(ones, self._camera_position)
            rtp_vel = np.ones((num_chunk_pixels, 6))
            rtp_vel[:, 0:3] = bl_point
            rtp_vel[:, 3:6] = momenta

            times = np.linspace(0, -10, self._iterations)

            ode_fcn = self._kerr_black_hole.ray_equation
            solutions, hit_horizon = self._rk4_final(
                ode_fcn,
                times,
                rtp_vel,
                carter_constant,
                self._kerr_black_hole.horizon_radius,
                spin,
                spin_sqr,
            )

            self._show_progress("generating sky layer...", process_num)

            vtheta = solutions[..., 1]
            vphi = solutions[..., 2]

            vuv = np.zeros((num_chunk_pixels, 2))

            vuv[:, 0] = np.mod(vphi, 2 * np.pi) / (2 * np.pi)
            vuv[:, 1] = np.mod(vtheta, np.pi) / (np.pi)

            colour = lookup(self._texture, vuv)[:, 0:3]

            self._show_progress("generating debug layers...", process_num)

            colour[hit_horizon] = np.zeros_like(colour)[hit_horizon]

            self._show_progress("beaming back to mothership.", process_num)

            if self._shuffle:
                self._colour_buffer_preproc[chunk] = colour
            else:
                self._colour_buffer_preproc[chunk[0]:(chunk[-1] + 1)] = colour

            self._show_progress("garbage collection...", process_num)
            gc.collect()

        self._show_progress("Done.", process_num)
