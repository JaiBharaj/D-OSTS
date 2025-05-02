class ExtendedKalmanFilter:
    def __init__(self, f_dynamics, F_jacobian, 
                 H, Q, R, x0, P0):
        """
        f_dynamics(x): returns dx/dt (4,)
        F_jacobian(x): returns Jacobian ∂f/∂x at x (4x4)
        H: measurement Jacobian (mx4)
        Q: process noise cov (4x4)
        R: measurement noise cov (mxm)
        x0: initial state (4,)
        P0: initial covariance (4x4)
        """
        self.f   = f_dynamics
        self.Fj  = F_jacobian
        self.H   = H
        self.Q   = Q
        self.R   = R
        self.x   = x0.copy()
        self.P   = P0.copy()

    def predict(self, dt):
        # State propagation (Euler currently)
        dx       = self.f(self.x)
        self.x  += dx * dt

        # Covariance propagation
        F_cont   = self.Fj(self.x)               # continuous Jacobian
        Phi      = np.eye(len(self.x)) + F_cont*dt
        self.P   = Phi @ self.P @ Phi.T + self.Q
        return self.x, self.P

    def update(self, z):
        # Measurement residual
        z_pred = self.H @ self.x
        y      = z - z_pred

        S      = self.H @ self.P @ self.H.T + self.R
        K      = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I      = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        return self.x, self.P

# Analytic Jacobian
def compute_F_analytic(x, CD, A, m, GM, rho_func):
    r, vr, theta, omega = x
    v_theta = r * omega
    v = np.hypot(vr, v_theta)
    rho = rho_func(r)
    D   = 0.5 * rho * CD * A / m

    # partials of v
    if v == 0:
        dv_dvr = dv_domega = dv_dr = 0.0
    else:
        dv_dvr    = vr / v
        dv_domega = r**2 * omega / v
        dv_dr     = (r * omega**2) / v

    F = np.zeros((4,4))
    # ∂dr/dt ∂x = [0,1,0,0]
    F[0,1] = 1

    # f1 = dvr/dt
    # ∂/∂r:    ω^2 + 2GM/r^3  − D·vr·dv_dr
    F[1,0] =       omega**2 + 2*GM/r**3 - D*vr*dv_dr
    # ∂/∂vr:  −D·(v + vr·dv_dvr)
    F[1,1] = - D*(v + vr*dv_dvr)
    # ∂/∂ω:    2rω − D·vr·dv_domega
    F[1,3] =   2*r*omega - D*vr*dv_domega

    # ∂dθ/dt/∂x = [0,0,0,1]
    F[2,3] = 1

    # f3 = dω/dt = -2 vr ω / r − D ω v
    # ∂/∂r:  2 vr ω/r^2  − D·ω·dv_dr
    F[3,0] =    2*vr*omega/r**2  - D*omega*dv_dr
    # ∂/∂vr: −2 ω/r  − D·ω·dv_dvr
    F[3,1] = - 2*omega/r - D*omega*dv_dvr
    # ∂/∂ω: −2 vr/r  − D·(v + ω·dv_domega)
    F[3,3] = - 2*vr/r - D*(v + omega*dv_domega)

    return F