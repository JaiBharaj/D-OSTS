from scipy.integrate import solve_ivp
from scipy.linalg import expm


class RK45Integrator:
    def step(self, f, x, dt):
        sol = solve_ivp(lambda t,y: f(y),
                        (0, dt), x,
                        rtol=1e-6, atol=1e-9,
                        max_step=dt/10)
        return sol.y[:, -1]

    def transition_matrix(self, x, dt):
        # continuous Jacobian at x (n×n)
        F_cont = compute_F_analytic(x, CD, A, m, GM, rho_func)
        # discrete transition matrix via matrix exponential
        return expm(F_cont * dt)

