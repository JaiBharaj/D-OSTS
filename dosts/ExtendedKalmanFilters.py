import numpy as np
from .CrudeInitialConditions import InitialConditions
from .CoordinateTransformations import spherical_to_cartesian


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with pluggable numerical integrator.

    The integrator must implement:
      - step(f, x, dt): returns next state x + dt * f(x)
      - transition_matrix(F_cont, dt): returns discrete-time approximation of transition matrix
    """

    def __init__(self, f_dynamics, f_jacobian, H, Q, R, x0, P0, integrator):
        """
        Args:
            f_dynamics : Callable(x) -> dx/dt
            f_jacobian : Callable(x) -> Jacobian matrix df/dx
            H : ndarray (m, n), measurement matrix
            Q : ndarray (n, n), process noise covariance
            R : ndarray (m, m), measurement noise covariance
            x0 : ndarray (n,), initial state
            P0 : ndarray (n, n), initial covariance
            integrator : object with step() and transition_matrix() methods
        """
        self.f = f_dynamics
        self.jac = f_jacobian
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()
        self.integrator = integrator
        self.pred_trajectory = {'t': [], 'r': [], 'theta': [], 'phi': []}
        self.uncertainty = {'r_var': [], 'theta_var': [], 'phi_var': [], 'measured': []}


    def predict(self, dt):
        """
        Predict step of EKF:
        1. Propagate state with integrator
        2. Compute Jacobian and state transition matrix
        3. Propagate covariance
        """
        # 1: state propagation
        self.x = self.integrator.step(self.f, self.x, dt)

        # 2: linearisation and transition matrix
        # F_cont = self.jac(self.x)
        F_disc = self.integrator.transition_matrix(self.x, dt)

        # 3: covariance propagation
        self.P = F_disc @ self.P @ F_disc.T + self.Q
        return self.x, self.P

    def update(self, z, eps=1e-8):
        # 1. innovation
        z_pred = self.H @ self.x
        y = z - z_pred

        # 2. innovation covariance

        S = self.H @ self.P @ self.H.T + self.R

        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        # 3. state update
        self.x = self.x + K @ y

        # Joseph form for P to maintain symmetry/PD
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T \
                 + K @ self.R @ K.T

        return self.x, self.P

    def predict_single_step(self, position, dt):
        x, P = self.predict(dt)
        x, P = self.update(position)
        return x, P

    def predict_trajectory(self, mode, times, measurements):

        self.pred_trajectory['t'] = times

        for i in range(len(times)):
            t = times[i]
            z = measurements[i]

            if i == 0:
                dt = 1e-3  # Small nonzero dt for first Jacobian estimate
            else:
                dt = t - times[i - 1]

            x, P = self.predict(dt)

            is_measured = False
            if not np.isnan(z).any():

                x, P = self.update(z)
                is_measured = True

            self.pred_trajectory['r'].append(x.copy()[0])
            self.pred_trajectory['theta'].append(x.copy()[2])
            self.uncertainty['r_var'].append(P.copy()[0,0])
            self.uncertainty['theta_var'].append(P.copy()[2, 2])
            self.uncertainty['measured'].append(is_measured)

            if mode.upper()=='3D':
                self.pred_trajectory['phi'].append(x.copy()[4])
                self.uncertainty['phi_var'].append(P.copy()[4, 4])

    def get_trajectory(self, mode):
        if mode.upper()=='2D':
            return np.array([self.pred_trajectory['t'], self.pred_trajectory['r'], self.pred_trajectory['theta']]).T
        else:
            return np.array([self.pred_trajectory['t'], self.pred_trajectory['r'],
                             self.pred_trajectory['theta'], self.pred_trajectory['phi'],
                             self.uncertainty['r_var'], self.uncertainty['theta_var'], self.uncertainty['phi_var'], self.uncertainty['measured']]).T

    def get_uncertainty(self, mode):
        if mode.upper()=='2D':
            return np.array([self.uncertainty['r_var'], self.uncertainty['theta_var'], self.uncertainty['measured']]).T
        else:
            return np.array([self.uncertainty['r_var'], self.uncertainty['theta_var'],
                           self.uncertainty['phi_var'], self.uncertainty['measured']])

    def crash(self, N=100, dt=1.0, max_steps=10000):
        """
        Monte Carlo prediction of crash impact angle (theta) using N samples.
        """
        crash_angles = []
        for j in range(N):
            sample = np.random.multivariate_normal(self.x, self.P)
            t = 0.0
            steps = 0
            while sample[0] > InitialConditions.earthRadius and steps < max_steps:
                sample = self.integrator.step(self.f, sample, dt)
                t += dt
                steps += 1
            if steps >= max_steps:
                print(f"[MC {j}] max_steps reached without crash.")
                continue
            crash_angles.append(sample[2])
            print(f"[MC {j}] crash at θ = {sample[2]:.6f} after {steps} steps")

        crash_angles = np.array(crash_angles)
        if len(crash_angles) > 0:
            print(f"Crash θ mean ± std: {np.mean(crash_angles):.6f} ± {np.std(crash_angles):.6f}")
        else:
            print("No crashes recorded.")
        return crash_angles

    def crash3D(self, N=100, dt=1.0, max_steps=10000):
        """
        Monte Carlo prediction of crash (theta, phi) using N samples in 3D.
        """
        crash_angles = []
        for j in range(N):
            sample = np.random.multivariate_normal(self.x, self.P)
            t = 0.0
            steps = 0
            while sample[0] > InitialConditions.earthRadius and steps < max_steps:
                sample = self.integrator.step(self.f, sample, dt)
                steps += 1
            if sample[0] > InitialConditions.earthRadius:
                print(f"[MC {j}] Max steps reached without crash.")
                continue
            crash_angles.append((sample[2], sample[4]))  # (theta, phi)
            print(f"[MC {j}] Crash at θ = {sample[2]:.6f}, φ = {sample[4]:.6f} after {steps} steps.")
        crash_angles = np.array(crash_angles)
        if len(crash_angles):
            thetas, phis = crash_angles[:, 0], crash_angles[:, 1]
            print(f"Crash θ mean ± std: {np.mean(thetas):.6f} ± {np.std(thetas):.6f}")
            print(f"Crash φ mean ± std: {np.mean(phis):.6f} ± {np.std(phis):.6f}")
        return crash_angles

    def crash3D_with_thrust(self, delta_v, h_thrust, N=100, dt=1.0, max_steps=10000):
        crash_angles = []
        for j in range(N):
            sample = np.random.multivariate_normal(self.x, self.P)
            t = 0.0
            steps = 0
            thrust_applied = False
            while sample[0] > InitialConditions.earthRadius and steps < max_steps:
                if not thrust_applied and sample[0] <= InitialConditions.earthRadius + h_thrust:
                    r, vr, phi, vphi, lam, vlam = sample
                    e_r = np.array(spherical_to_cartesian(1, lam, phi))
                    e_phi = np.array(spherical_to_cartesian(1, lam, phi + np.pi / 2))
                    e_lam = np.array(spherical_to_cartesian(1, lam + np.pi / 2, np.pi / 2))
                    v_vec = vr * e_r + r * vphi * e_phi + r * np.sin(phi) * vlam * e_lam
                    v_hat = v_vec / np.linalg.norm(v_vec)
                    v_vec_new = v_vec + delta_v * v_hat
                    sample[1] = np.dot(v_vec_new, e_r)
                    sample[3] = np.dot(v_vec_new, e_phi) / r
                    sample[5] = np.dot(v_vec_new, e_lam) / (r * np.sin(phi))
                    thrust_applied = True
                sample = self.integrator.step(self.f, sample, dt)
                steps += 1
            if sample[0] > InitialConditions.earthRadius:
                print(f"[MC {j}] Max steps reached without crash (with thrust).")
                continue
            crash_angles.append((sample[2], sample[4]))
            print(f"[MC {j}] Crash with thrust at θ = {sample[2]:.6f}, φ = {sample[4]:.6f} after {steps} steps.")
        crash_angles = np.array(crash_angles)
        if len(crash_angles):
            thetas, phis = crash_angles[:, 0], crash_angles[:, 1]
            print(f"Crash (with thrust) θ mean ± std: {np.mean(thetas):.6f} ± {np.std(thetas):.6f}")
            print(f"Crash (with thrust) φ mean ± std: {np.mean(phis):.6f} ± {np.std(phis):.6f}")
        return crash_angles


def compute_F_analytic(x, CD, A, m, GM, rho_func):
    """
    Computes the analytic Jacobian matrix F = ∂f/∂x for polar orbital dynamics with drag.

    Parameters
    ----------
    x : ndarray (4,)
        State vector [r, vr, theta, omega]
    CD : float
        Drag coefficient
    A : float
        Cross-sectional area of satellite (m^2)
    m : float
        Mass of satellite (kg)
    GM : float
        Gravitational constant x Earth mass (m^3/s^2)
    rho_func : Callable
        Function rho_func(r): returns atmospheric density at radius r

    Returns
    -------
    F : ndarray (4, 4)
        Jacobian matrix of the dynamics evaluated at x
    """
    r, vr, theta, omega = x

    # Velocity components
    v_theta = r * omega
    v = np.hypot(vr, v_theta)

    # Drag factor
    rho = rho_func(r)
    D = 0.5 * rho * CD * A / m

    # Partial derivatives of v
    if v == 0:
        dv_dvr = 0.0
        dv_domega = 0.0
        dv_dr = 0.0
    else:
        dv_dvr = vr / v
        dv_domega = r ** 2 * omega / v
        dv_dr = r * omega ** 2 / v

    # Jacobian matrix
    F = np.zeros((4, 4))

    # ∂(dr/dt)/∂x = [0, 1, 0, 0]
    F[0, 1] = 1.0

    # ∂(dvr/dt)/∂x
    F[1, 0] = omega ** 2 + 2 * GM / r ** 3 - D * vr * dv_dr
    F[1, 1] = - D * (v + vr * dv_dvr)
    F[1, 3] = 2 * r * omega - D * vr * dv_domega

    # ∂(dtheta/dt)/∂x = [0, 0, 0, 1]
    F[2, 3] = 1.0

    # ∂(domega/dt)/∂x
    F[3, 0] = 2 * vr * omega / r ** 2 - D * omega * dv_dr
    F[3, 1] = -2 * omega / r - D * omega * dv_dvr
    F[3, 3] = -2 * vr / r - D * (v + omega * dv_domega)

    return F


def compute_F_spherical(x, CD, A, m, GM, rho_func):
    r, vr, theta, omega_theta, phi, omega_phi = x
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin2 = sin_theta ** 2
    cos_sin = cos_theta * sin_theta

    rho = rho_func(r - InitialConditions.earthRadius)
    D = 0.5 * rho * CD * A / m

    v_theta = r * omega_theta
    v_phi = r * sin_theta * omega_phi
    v_sq = vr ** 2 + v_theta ** 2 + v_phi ** 2
    v = np.sqrt(v_sq) if v_sq > 1e-8 else 1e-8

    # ∂v/∂x
    dv = np.zeros(6)
    dv[0] = (v_theta * omega_theta + v_phi * omega_phi * sin_theta) * r / v
    dv[1] = vr / v
    dv[2] = (v_phi * omega_phi * cos_theta) * r / v
    dv[3] = r ** 2 * omega_theta / v
    dv[5] = r ** 2 * sin2 * omega_phi / v

    F = np.zeros((6, 6))
    F[0, 1] = 1.0
    # dvr/dt = r*(ωθ^2 + sin^2θ * ωφ^2) - GM/r^2 - D * vr * v
    F[1, 0] = omega_theta ** 2 + sin2 * omega_phi ** 2 + 2 * GM / r ** 3 - D * vr * dv[0]
    F[1, 1] = -D * (v + vr * dv[1])
    F[1, 2] = 2 * sin_theta * cos_theta * omega_phi ** 2 - D * vr * dv[2]
    F[1, 3] = 2 * r * omega_theta - D * vr * dv[3]
    F[1, 5] = 2 * r * sin2 * omega_phi - D * vr * dv[5]

    # dθ/dt = ωθ
    F[2, 3] = 1.0

    # dωθ/dt = -2*vr*ωθ/r + sinθ*cosθ*ωφ^2 - D*ωθ*v
    F[3, 0] = 2 * vr * omega_theta / r ** 2 - D * omega_theta * dv[0]
    F[3, 1] = -2 * omega_theta / r - D * omega_theta * dv[1]
    F[3, 2] = cos_sin * omega_phi ** 2 - D * omega_theta * dv[2]
    F[3, 3] = -2 * vr / r - D * (v + omega_theta * dv[3])
    F[3, 5] = 2 * cos_sin * omega_phi - D * omega_theta * dv[5]

    # dφ/dt = ωφ
    F[4, 5] = 1.0

    # dωφ/dt = -2*vr*ωφ/r - 2*(ωθ/ tanθ)*ωφ - D*ωφ*v
    F[5, 0] = 2 * vr * omega_phi / r ** 2 - D * omega_phi * dv[0]
    F[5, 1] = -2 * omega_phi / r - D * omega_phi * dv[1]
    F[5, 3] = -2 * omega_phi / np.tan(theta) - D * omega_phi * dv[3]
    F[5, 5] = -2 * vr / r - 2 * omega_theta / np.tan(theta) - D * (v + omega_phi * dv[5])

    # safe denominator for the 1/sin²θ term:
    sin2_safe = max(sin2, 1e-8)

    # ∂(dωφ/dt)/∂θ
    F[5, 2] = 2 * omega_theta * omega_phi / sin2_safe \
              - D * omega_phi * dv[2]

    return F