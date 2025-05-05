from scipy.integrate import solve_ivp
from scipy.linalg import expm
from ExtendedKalmanFilters import compute_F_analytic

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
