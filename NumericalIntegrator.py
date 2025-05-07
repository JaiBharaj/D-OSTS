import numpy as np
from CrudeInitialConditions import InitialConditions
from CoordinateTransformations import PolarAccelerations, SphericalAccelerations
from scipy.integrate import solve_ivp

class Integrator:

    def __init__(self):
        self.Re = InitialConditions.earthRadius
        self.alt0 = InitialConditions.initSatAlt
        self.r0 = self.Re + self.alt0
        self.theta0 = InitialConditions.initSatTheta
        self.phi0 = InitialConditions.initSatPhi
        self.lam0 = InitialConditions.initSatLam
        self.r_dot0 = InitialConditions.initSatRdot
        self.phi_dot0 = InitialConditions.initSatPhidot
        self.mu = InitialConditions.gravConstant * InitialConditions.earthMass

        v_circ = np.sqrt(self.mu / self.r0)
        v_target = v_circ - 80.0
        term_sq = v_target ** 2 - (self.r0 * self.phi_dot0) ** 2
        if term_sq < 0:
            raise ValueError("phi_dot0 too large for given Δv or orbit height")
        self.lam_dot0 = - np.sqrt(term_sq) / (self.r0 * np.sin(self.phi0))

    def ode_system(t, u):
        u = np.asarray(u).flatten()
        if len(u) != 4:
            raise ValueError(f"Expected 4D state vector, got {len(u)} elements")
        u1, u2, u3, u4 = u
        return PolarAccelerations.accelerations(u1, u2, u3, u4)

    def transition_matrix(self, u0, h=0.01, epsilon=1e-6):
        u0 = np.asarray(u0).flatten()  # Ensure 1D
        n = len(u0)
        M = np.zeros((n, n))

        # Integrate base trajectory
        sol_base = solve_ivp(self.ode_system, (0, h), u0, method='RK45', t_eval=[h])
        u_base = np.array(sol_base.y)[:, -1]

        for i in range(n):
            u_perturbed = u0.copy()
            u_perturbed[i] += epsilon

            sol_perturbed = solve_ivp(self.ode_system, (0, h), u_perturbed, method='RK45', t_eval=[h])
            u_i = sol_perturbed.y[:, -1]

            M[:, i] = (u_i - u_base) / epsilon

        return M

    def step(f, x, dt):
        return x + dt * f(x)

    @staticmethod
    def rhs_polar(t, y):
        r, r_dot, theta, theta_dot = y
        return PolarAccelerations.accelerations(r, r_dot, theta, theta_dot)

    @staticmethod
    def rhs_spherical(t, y):
        r, r_dot, phi, phi_dot, lam, lam_dot = y
        return SphericalAccelerations.accelerations(
            r, r_dot, phi, phi_dot, lam, lam_dot)
    
    def hit_ground(self):
        def events(t, y):
            return y[0] - self.Re

        events.terminal = True
        events.direction = -1
        return events

    def runge_kutta45_2d(self):
        v0 = np.sqrt(self.mu / self.r0)
        theta_dot0 = (v0 - 80) / self.r0

        y0 = np.array([
            self.r0,
            self.r_dot0,
            self.theta0,
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
            events=self.hit_ground())

        return sol
    
    def runge_kutta45_3d(self):
        

        y0 = np.array([
            self.r0,      # r
            self.r_dot0,  # ṙ
            self.phi0,    # φ
            self.phi_dot0,# φ̇
            self.lam0,    # λ
            self.lam_dot0            # λ̇
        ], dtype=float)

        sol = solve_ivp(
            Integrator.rhs_spherical,
            (0.0, 1.3e8),
            y0,
            method="RK45",
            rtol=1e-7, 
            atol=1e-9,
            first_step=1.0, 
            max_step=1.5,
            events=self.hit_ground())

        return sol

    def downsample_indices(n_total, n_target):
        if n_target >= n_total:
            return np.arange(n_total)
        return np.linspace(0, n_total - 1, n_target, dtype=int)

    def get_trajectory(self, n_save=20_000):
        res = self.runge_kutta45_2d()
        print(res.t_events[0])

        t_arr = res.t
        r_arr = res.y[0]
        th_arr = res.y[2]

        # -------- Optional ----------
        # keep = self.downsample_indices(len(t_arr), n_save)
        # t_arr, r_arr, th_arr = t_arr[keep], r_arr[keep], th_arr[keep]

        # -------- File output ----------
        true_file = 'trajectory_without_noise.txt'

        with open(true_file, 'w') as f_true:
            for (t, r, th) in zip(t_arr, r_arr, th_arr):
                f_true.write(f"{t} {r} {th}\n")

    def get_trajectory_3d(self, n_save=20_000, to_xyz=False):
        res = self.runge_kutta45_3d()
        t_arr  = res.t
        r_arr  = res.y[0]
        phi_arr = res.y[2]
        lam_arr = res.y[4]

        # ------ option ------
        # keep = self.downsample_indices(len(t_arr), n_save)
        # t_arr, r_arr, phi_arr, lam_arr = t_arr[keep], r_arr[keep], phi_arr[keep], lam_arr[keep]

        fname = 'trajectory_without_noise_3d_xyz.txt' if to_xyz else 'trajectory_without_noise_3d.txt'
        #with open(fname, 'w') as f, open('trajectory_3d_xyz_pred.txt', 'w') as f_pred:
        with open(fname, 'w') as f:
            if to_xyz:
                x = r_arr * np.sin(phi_arr) * np.cos(lam_arr)
                y = r_arr * np.sin(phi_arr) * np.sin(lam_arr)
                z = r_arr * np.cos(phi_arr)
                for t, xi, yi, zi in zip(t_arr, x, y, z):
                    f.write(f"{t:.3f} {xi:.3f} {yi:.3f} {zi:.3f}\n")
                    # f.write(f"{xi:.3f} {yi:.3f} {zi:.3f}\n")
                    # f_pred.write(f"{0} {0} {0} {0} {0} {0} {0}\n")
            else:
                for t, r, ph, la in zip(t_arr, r_arr, phi_arr, lam_arr):
                    f.write(f"{t:.3f} {r:.3f} {ph:.6f} {la:.6f}\n")

        print(f"Wrote {len(t_arr)} points to {fname}")


# ------ An example to use this class ------

# from NumericalIntegrator import Integrator
# from CrudeInitialConditions import InitialConditions

# integrator = Integrator()
# integrator.get_trajectory_3d()

# ------ Then you will get a text file named 'trajectory_without_noise_3d.txt' in same path ------
