# Predictor Integrator

This module defines integrators for use in predictor-corrector frameworks. It includes three classes: 
- `RK45Integrator`: Integrator for 2D (polar) dynamics, supporting both one-step RK45 integration and analytic Jacobian matrix computation for use in Kalman Filters.
- `RK45Integrator_3D`: Same as `RK45Integrator`, but for full 3D spherical coordinate dynamics using a different Jacobian function.
- `Integrator3D`: Subclass of the main simulation `Integrator`, extended to allow Jacobian estimation via finite differences for Kalman filters or linear approximations.

---

```python
def __init__(self, CD, A, m, GM, rho_func):
    ...
```
- **Parameters**:
    - `CD`: Drag coefficient.
    - `A`: Cross-sectional area of the satellite.
    - `m`: Satellite mass.
    - `GM`: Gravitational parameter (G·Mₑ).
    - `rho_func`: A function that returns atmospheric density given altitude or radius.

---

```python
def step(self, f, x, dt):
    sol = solve_ivp(lambda t, y: f(y),
                    (0, dt), x,
                    rtol=1e-6, atol=1e-9,
                    max_step=dt / 10)
    return sol.y[:, -1]
```
- **Purpose**: Performs a single Runge-Kutta 45 (RK45) integration step.
- **Parameters**:
    - `f`: Dynamics function taking a state vector `x` and returning its time derivative.
    - `x`: Current state vector.
    - `dt`: Time increment for integration.
- **Returns**: State vector after time `dt`.

---

```python
def transition_matrix(self, x, dt):
    ...
    return expm(F_cont * dt)
```
- **Purpose**: Computes the continuous-time Jacobian matrix at state `x`, then converts it to a discrete-time transition matrix using the matrix exponential.
- **Returns**: Discrete transition matrix `Φ = exp(F·dt)` using `compute_F_analytic()` or `compute_F_spherical()` from `ExtendedKalmanFilters`.

---

```python
def step(self, f, x0, dt):
    # use the same tolerances & max_step as run_rk45_3d
    sol = solve_ivp(
        Integrator.rhs_spherical,
        (0.0, dt),
        x0,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        max_step=1.5
    )
    return sol.y[:, -1]
```
- **Purpose**: Integrates the state vector `x0` forward by `dt` seconds using RK45.
- **Note**: The `f` parameter is ignored and hardcoded to `Integrator.rhs_spherical`.

---

```python
def transition_matrix(self, x0, dt, eps=1e-6):
    # finite-difference Jacobian over dt
    n = len(x0)
    M = np.zeros((n, n))
    base = self.step(None, x0, dt)
    for i in range(n):
        xp = x0.copy()
        xp[i] += eps
        pert = self.step(None, xp, dt)
        M[:, i] = (pert - base) / eps
    return M
```
- **Purpose**: Numerically approximates the Jacobian matrix ∂f/∂x using finite differences.
- **Parameters**:
    - `x0`: State vector at which to evaluate the Jacobian.
    - `dt`: Integration duration.
    - `eps`: Perturbation size for finite difference.
