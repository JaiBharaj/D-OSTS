# Model Dynamics

This module defines two classes used to compute the accelerations of a satellite in polar and spherical coordinate systems. \
These accelerations are used in numerical solvers such as RK45 for trajectory simulation.
- `PolarAccelerations`: Provides acceleration functions in polar coordinates.
- `SphericalAccelerations`: Inherits from `PolarAccelerations` and extends the model to spherical coordinates (radial, polar, and azimuthal), enabling 3D orbital dynamics modeling.

---

```python
@staticmethod
def comb_velocity(rad, radVel, angVel):
    return np.sqrt(radVel**2 + (rad * angVel)**2)
```
- **Purpose**: Computes the total velocity magnitude in polar coordinates from radial and angular components.
- **Parameters**:
    - `rad`: Radial position (distance from Earth's center), in meters.
    - `radVel`: Radial velocity (ṙ), in m/s.
    - `angVel`: Angular velocity (θ̇), in rad/s.
- **Implementation**:
    - Computes total velocity as the Euclidean norm of radial and tangential velocity: √(ṙ² + (rθ̇)²).

---

```python
@staticmethod
def drag_start(C_d, A, m, r):
    Re = 6371e3
    h = r - Re
    result = 0.5 * atmospheric_density(h) * C_d * A / m
    return result
```
- **Purpose**: Calculates the atmospheric drag force.
- **Parameters**:
    - `C_d`: Drag coefficient (dimensionless).
    - `A`: Cross-sectional area of the satellite (in m²).
    - `m`: Mass of the satellite (in kg).
    - `r`: Radial position from Earth’s center (in m).
- **Implementation**:
    - Computes the altitude above Earth’s surface: `h = r - Re`.
    - Queries the atmospheric density at altitude `h`.
    - Returns the constant multiplier part of the drag force expression.

---

```python
@staticmethod
def accelerations(u1, u2, u3, u4):
    ...
    return np.array([u1_dot, u2_dot, u3_dot, u4_dot])
```
- **Purpose**: Computes the time derivatives of the satellite’s state in polar coordinates.
- **Parameters**:
    - `u1`: Radial position (r)
    - `u2`: Radial velocity (ṙ)
    - `u3`: Angular position (θ)
    - `u4`: Angular velocity (θ̇)
- **Implementation**:
    - Computes the drag using `drag_start()`.
    - Computes total velocity with `comb_velocity()`.
    - Uses Newton's second law and angular motion equations to derive accelerations.
    - Returns `[ṙ, r̈, θ̇, θ̈]`.

---

```python
@staticmethod
def comb_velocity(rad, radVel, polAng, polVel, aziVel):
    return np.sqrt(
        radVel**2 + rad**2 * (polVel**2 + np.sin(polAng)**2 * aziVel**2))
```
- **Purpose**: Computes the total velocity magnitude in spherical coordinates using radial, polar, and azimuthal components.
- **Parameters**:
    - `rad`: Radial distance.
    - `radVel`: Radial velocity.
    - `polAng`: Polar angle.
    - `polVel`: Polar angular velocity.
    - `aziVel`: Azimuthal angular velocity.
- **Implementation**:
    - Calculates velocity magnitude using the spherical velocity formula:\
    √(ṙ² + r²(φ̇² + sin²(φ)·λ̇²)).

---

```python
@staticmethod
def accelerations(u1, u2, u3, u4, u5, u6):
    ...
    return np.array([u1_dot, u2_dot, u3_dot, u4_dot, u5_dot, u6_dot])
```
- **Purpose**: Computes the full state derivatives of a satellite in spherical coordinates, suitable for high-fidelity simulations.
- **Parameters**:
    - `u1`: Radial position (r)
    - `u2`: Radial velocity (ṙ)
    - `u3`: Polar angle (φ)
    - `u4`: Polar angular velocity (φ̇)
    - `u5`: Azimuthal angle (λ)
    - `u6`: Azimuthal angular velocity (λ̇)
- **Implementation**:
    - Calculates drag coefficient using `drag_start()`.
    - Computes velocity magnitude with `comb_velocity()`.
    - Applies equations of motion in spherical coordinates.
    - Returns `[ṙ, r̈, φ̇, φ̈, λ̇, λ̈]`.
