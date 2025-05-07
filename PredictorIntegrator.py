from scipy.integrate import solve_ivp
from scipy.linalg import expm
from ExtendedKalmanFilters import compute_F_analytic, compute_F_spherical
from NumericalIntegrator import Integrator
import numpy as np

class RK45Integrator:
    def __init__(self, CD, A, m, GM, rho_func):
        self.CD = CD
        self.A = A
        self.m = m
        self.GM = GM
        self.rho_func = rho_func

    def step(self, f, x, dt):
        sol = solve_ivp(lambda t, y: f(y),
                        (0, dt), x,
                        rtol=1e-6, atol=1e-9,
                        max_step=dt/10)
        return sol.y[:, -1]

    def transition_matrix(self, x, dt):
        F_cont = compute_F_analytic(x, self.CD, self.A, self.m, self.GM, self.rho_func)
        return expm(F_cont * dt)

class RK45Integrator_3D:
    def __init__(self, CD, A, m, GM, rho_func):
        self.CD = CD
        self.A = A
        self.m = m
        self.GM = GM
        self.rho_func = rho_func

    def step(self, f, x, dt):
        sol = solve_ivp(lambda t, y: f(y),
                        (0, dt), x,
                        rtol=1e-6, atol=1e-9,
                        max_step=dt/10)
        return sol.y[:, -1]

    def transition_matrix(self, x, dt):
        F_cont = compute_F_spherical(x, self.CD, self.A, self.m, self.GM, self.rho_func)
        return expm(F_cont * dt)



class Integrator3D(Integrator):
    def step(self, f, x0, dt):
        # use the same tolerances & max_step as run_rk45_3d
        sol = solve_ivp(
            Integrator.rhs_spherical,
            (0.0, dt),
            x0,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            first_step=1.0,
            max_step=1.5
        )
        return sol.y[:, -1]

    def transition_matrix(self, x0, dt, eps=1e-6):
        # finite-difference Jacobian over dt
        n = len(x0)
        M = np.zeros((n, n))
        base = self.step(None, x0, dt)
        for i in range(n):
            xp = x0.copy(); xp[i] += eps
            pert = self.step(None, xp, dt)
            M[:, i] = (pert - base) / eps
        return M
