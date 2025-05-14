# Numerical Integrator

This module defines the `Integrator` class, which performs numerical integration of satellite trajectories using the Runge-Kutta 45 method (RK45) from `solve_ivp`. It supports both 2D (polar) and 3D (spherical) formulations, includes ground impact detection, and models optional mid-flight thrust.

---

```python
def __init__(self, recorded_times=None):
```
- **Purpose**: Initializes the satellite's state vector, orbital constants, and optional event times for trajectory recording.
- **Parameters**:
    - `recorded_times`: Array of times at which to record the solution.
- **Implementation**:
    - Sets initial orbital elements from `InitialConditions`.
    - Computes initial azimuthal velocity `lam_dot0` to satisfy circular orbit mechanics minus deltaV.
    - Raises an error if the initial configuration is not physically valid.

---

```python
@staticmethod
def rhs_polar(t, y):
    r, r_dot, theta, theta_dot = y
    return PolarAccelerations.accelerations(r, r_dot, theta, theta_dot)
```
- **Purpose**: Right-hand side (derivatives) of the ODE system for 2D polar motion.
- **Parameters**:
    - `t`: Time (not used).
    - `y`: State vector `[r, ṙ, θ, θ̇]`.
- **Implementation**:
    - Returns `[ṙ, r̈, θ̇, θ̈]` from the `PolarAccelerations` model.

---

```python
@staticmethod
def rhs_spherical(t, y):
    r, r_dot, phi, phi_dot, lam, lam_dot = y
    return SphericalAccelerations.accelerations(r, r_dot, 
    phi, phi_dot, lam, lam_dot)
```
- **Purpose**: Right-hand side (derivatives) of the ODE system for 3D spherical motion.
- **Parameters**:
    - `t`: Time (not used).
    - `y`: State vector `[r, ṙ, φ, φ̇, λ, λ̇]`.
- **Implementation**:
    - Returns `[ṙ, r̈, φ̇, φ̈, λ̇, λ̈]` from the `SphericalAccelerations` model.

---

```python
def hit_ground(self):
    ...
    return events
```
- **Purpose**: Event function for RK45 to detect when the satellite hits Earth's surface.
- **Implementation**:
    - Triggers termination when `r - Re = 0` (i.e., satellite radius equals Earth’s radius).

---

```python
def at_thrust(self):
    ...
    return events
```
- **Purpose**: Event function that detects when the satellite reaches a specific altitude (`h_thrust`) where thrust is applied.
- **Implementation**:
    - Triggers termination when `r = Re + h_thrust`.

---

```python
def runge_kutta45_2d(self):
    ...
    return sol
```
- **Purpose**: Solves the 2D polar equations of motion using RK45 integration.
- **Implementation**:
    - Uses `rhs_polar` for orbit simulation.
    - Integrates until ground impact using `solve_ivp`, and return the solution.

---

```python
def runge_kutta45_3d(self):
    ...
    return sol
```
- **Purpose**: Solves the 3D spherical equations of motion using RK45 integration.
- **Implementation**:
    - Uses `self.y0` and `rhs_spherical` for full orbit simulation.
    - Integrates until ground impact using `solve_ivp`, and return the solution.

---

```python
@staticmethod
def great_circle_distance(phi1, lam1, phi2, lam2):
    Δσ = np.arccos(
        np.sin(phi1) * np.sin(phi2) +
        np.cos(phi1) * np.cos(phi2) * np.cos(lam1 - lam2)
    )
    return IC.earthRadius * Δσ
```
- **Purpose**: Computes the great-circle distance between two points on the Earth.
- **Parameters**:
    - `phi1, lam1`: Latitude and longitude of point 1.
    - `phi2, lam2`: Latitude and longitude of point 2.
- **Implementation**: Uses spherical law of cosines

---

```python
@staticmethod
def in_populated(phi, lam):
    for pc in IC.populatedCenters:
        if Integrator.great_circle_distance(phi, lam, pc[0], pc[1]) < 
            IC.populatedRadius:
            return True
    return False
```
- **Purpose**: Determines whether a given coordinate lies within any of the predefined populated zones.
- **Parameters**:
    - `phi, lam`: Target coordinates.
- **Implementation**:
    - Loops through all populated centers and checks distance against `populatedRadius`.

---

```python
def get_trajectory_2d(self):
    ...
    return np.array([t_arr, r_arr, th_arr]).T
```
- **Purpose**: Integrates 2D trajectory and returns sampled output values `[time, radius, theta]`.

---

```python
def get_trajectory_3d(self, bonus=False):
    ...
    return np.array([t_arr, r_arr, lam_arr, phi_arr]).T
```
- **Purpose**: Integrates the 3D trajectory and applies optional thrust manually or if target would impact populated area.
- **Parameters**: 
    - `bonus`: Set the thrust manually.
- **Implementation**:
    - Runs RK45 integration to ground and get the crash position.
    - If impact site is populated or bonus is enabled, simulates thrust:
        - Integrates to thrust altitude.
        - Applies deltaV vectorially to velocity.
        - Continues integration to ground.
        - Merges both trajectory segments into one.
    - Extracts and returns trajectory `[time, radius, azimuthal_angle, polar_angle]`.
