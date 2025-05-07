import numpy as np
from CrudeInitialConditions import InitialConditions

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
        y      = z - z_pred

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

def compute_F_spherical(x, CD, A, m, GM, rho_func):
    r, vr, theta, omega_theta, phi, omega_phi = x
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin2 = sin_theta**2
    cos_sin = cos_theta * sin_theta

    rho = rho_func(r - InitialConditions.earthRadius)
    D   = 0.5 * rho * CD * A / m

    v_theta = r * omega_theta
    v_phi   = r * sin_theta * omega_phi
    v_sq    = vr**2 + v_theta**2 + v_phi**2
    v       = np.sqrt(v_sq) if v_sq>1e-8 else 1e-8

    # ∂v/∂x
    dv = np.zeros(6)
    dv[0] = (v_theta*omega_theta + v_phi*omega_phi*sin_theta) * r / v
    dv[1] = vr / v
    dv[2] = (v_phi * omega_phi * cos_theta) * r / v
    dv[3] = r**2 * omega_theta / v
    dv[5] = r**2 * sin2 * omega_phi / v

    F = np.zeros((6,6))
    F[0,1] = 1.0
    # dvr/dt = r*(ωθ^2 + sin^2θ * ωφ^2) - GM/r^2 - D * vr * v
    F[1, 0] = omega_theta**2 + sin2 * omega_phi**2 + 2*GM / r**3 - D * vr * dv[0]
    F[1, 1] = -D * (v + vr * dv[1])
    F[1, 2] = 2 * sin_theta * cos_theta * omega_phi**2 - D * vr * dv[2]
    F[1, 3] = 2 * r * omega_theta            - D * vr * dv[3]
    F[1, 5] = 2 * r * sin2 * omega_phi       - D * vr * dv[5]

    # dθ/dt = ωθ
    F[2, 3] = 1.0

    # dωθ/dt = -2*vr*ωθ/r + sinθ*cosθ*ωφ^2 - D*ωθ*v
    F[3, 0] = 2 * vr * omega_theta / r**2 - D * omega_theta * dv[0]
    F[3, 1] = -2 * omega_theta / r          - D * omega_theta * dv[1]
    F[3, 2] = cos_sin * omega_phi**2      - D * omega_theta * dv[2]
    F[3, 3] = -2 * vr / r                  - D * (v + omega_theta * dv[3])
    F[3, 5] = 2 * cos_sin * omega_phi      - D * omega_theta * dv[5]

    # dφ/dt = ωφ
    F[4, 5] = 1.0

    # dωφ/dt = -2*vr*ωφ/r - 2*(ωθ/ tanθ)*ωφ - D*ωφ*v
    F[5, 0] = 2 * vr * omega_phi / r**2    - D * omega_phi * dv[0]
    F[5, 1] = -2 * omega_phi / r            - D * omega_phi * dv[1]
    F[5, 3] = -2 * omega_phi / np.tan(theta)     - D * omega_phi * dv[3]
    F[5, 5] = -2 * vr / r - 2 * omega_theta / np.tan(theta) - D * (v + omega_phi * dv[5])

    # safe denominator for the 1/sin²θ term:
    sin2_safe = max(sin2, 1e-8)

    # ∂(dωφ/dt)/∂θ
    F[5,2] = 2 * omega_theta * omega_phi / sin2_safe \
             - D * omega_phi * dv[2]
    
    return F

"""
Usage:

f_dynamics function is just PolarAccelerations.accelerations(*x)

f_jacobian is just compute_F_analytic from above

Needs the integrator to take step forward, this will be determinded by the one we use.
The integrator class within the predictor should have two functions:
1. step (RK45)
2. transition matrix (Mapping the errors from k to k+1)
    This can be done as follows:
    def compute_phi(self, F_cont, dt):
        return expm(F_cont * dt)

Then need to apply it to real data.
This will require simulation data and initial conditions for state (x) P,Q, R and H /
 which will have to be determined as we go.

# Measurement model
H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

# Measurement noise
sigma_r_meas = 5.0        # m
sigma_theta_meas = 1e-4   # rad
R = np.diag([sigma_r_meas**2, sigma_theta_meas**2])

# Process noise (initially zero)
Q = np.zeros((4, 4))

# Initial uncertainty
P0 = np.diag([
    10.0**2,        # r
    1.0**2,         # vr
    (1e-4)**2,      # theta
    (1e-4)**2       # omega
])


ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    integrator=EulerIntegrator()
)

for t in range(num_steps):
    ekf.predict(dt)
    z = generate_measurement(x_true[t])
    ekf.update(z)
    save(ekf.x, ekf.P)
"""
