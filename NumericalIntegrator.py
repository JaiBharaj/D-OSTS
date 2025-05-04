import numpy as np
from CrudeInitialConditions import InitialConditions
from CoordinateTransformations import PolarAccelerations
from scipy.integrate import solve_ivp

class Integrator:
    Re = InitialConditions.earthRadius
    alt0 = InitialConditions.initSatAlt
    r0 = Re + alt0
    theta0 = 0.0
    r_dot0 = 0.0
    mu = None

    @staticmethod
    def initialize(mu_value):
        Integrator.mu = mu_value

    @staticmethod
    def ode_system(t, u):
        u1, u2, u3, u4 = u
        return PolarAccelerations(u1, u2, u3, u4)

    @staticmethod
    def transmission_matrix(u0, h=0.01, epsilon=1e-6):
        n = len(u0)
        M = np.zeros((n, n))

        # Integrate base trajectory
        sol_base = solve_ivp(Integrator.ode_system, (0, h), u0, method='RK45', t_eval=[h])
        u_base = sol_base.y[:, -1]

        for i in range(n):
            u_perturbed = u0.copy()
            u_perturbed[i] += epsilon

            sol_perturbed = solve_ivp(Integrator.ode_system, (0, h), u_perturbed, method='RK45', t_eval=[h])
            u_i = sol_perturbed.y[:, -1]

            # Finite difference column
            M[:, i] = (u_i - u_base) / epsilon

        return M

    @staticmethod
    def rhs_polar(t, y):
        r, r_dot, theta, theta_dot = y
        return PolarAccelerations.accelerations(r, r_dot, theta, theta_dot)

    @staticmethod
    def hit_ground():
        def events(t, y):
            return y[0] - Integrator.Re

        events.terminal = True
        events.direction = -1
        return events

    @staticmethod
    def runge_kutta45_2d():
        v0 = np.sqrt(Integrator.mu / Integrator.r0)
        theta_dot0 = (v0 - 80) / Integrator.r0

        y0 = np.array([
            Integrator.r0,
            Integrator.r_dot0,
            Integrator.theta0,
            theta_dot0], dtype=float)

        sol = solve_ivp(
            Integrator.rhs_polar,
            (0.0, 1.3e8),
            y0,
            rtol=1e-6,
            atol=1e-8,
            method="RK45",
            first_step=1.0,
            max_step=5.0,
            events=Integrator.hit_ground())

        return sol

    @staticmethod
    def downsample_indices(n_total, n_target):
        if n_target >= n_total:
            return np.arange(n_total)
        return np.linspace(0, n_total - 1, n_target, dtype=int)

    @staticmethod
    def get_trajectory(n_save=20_000):
        res = Integrator.runge_kutta45_2d()
        print(res.t_events[0])

        t_arr = res.t
        r_arr = res.y[0]
        th_arr = res.y[2]

        # -------- Optional ----------
        # keep = Integrator.downsample_indices(len(t_arr), n_save)
        # t_arr, r_arr, th_arr = t_arr[keep], r_arr[keep], th_arr[keep]

        # -------- File output ----------
        true_file = 'trajectory_without_noise.txt'

        with open(true_file, 'w') as f_true:
            for (t, r, th) in zip(t_arr, r_arr, th_arr):
                f_true.write(f"{t} {r} {th}\n")

# ------ An example to use this class ------

# from NumericalIntegrator import Integrator
# from CrudeInitialConditions import InitialConditions

# Integrator.initialize(mu_value=InitialConditions.gravConstant * InitialConditions.earthMass)
# Integrator.get_trajectory()

# ------ Then you will get a text file named 'trajectory_without_noise' in same path ------