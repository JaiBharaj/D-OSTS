# Radar

This module defines functions and a class to simulate radar station deployment and measurements for tracking a satellite. It supports both 2D (polar) and 3D (spherical) coordinate systems, including radar visibility logic, noise modeling, and measurement aggregation.

---

```python
def distribute_radars2D(Hdark, R_earth):
    ...
    return radar_coors
```
- **Purpose**: Distributes 2D radar stations evenly around the Earth's surface based on visibility constraints.
- **Parameters**:
  - `Hdark`: Maximum measurable satellite altitude in meters.
  - `R_earth`: Radius of the Earth.
- **Returns**: Array of shape `(N, 2)` with radar positions `[r, θ]`.

---

```python
def fibonacci_sphere_grid(N):
    gold = 0.5 * (1 + np.sqrt(5))
    ind = np.array([i for i in range(0, N)])
    x = (ind / gold) % gold
    y = ind / (N - 1)
    theta = 2 * np.pi * x
    phi = np.arccos(1 - 2 * y)
    return theta, phi
```
- **Purpose**: Generates evenly distributed points on a sphere using a Fibonacci lattice.
- **Parameters**:
  - `N`: Number of points to generate.
- **Returns**: Arrays of spherical coordinates `(θ, φ)`.

---

```python
def distribute_radars3D(H_dark, R_Earth):
    ...
    return np.column_stack((np.full(num_stations, R_Earth), phi, theta))
```
- **Purpose**: Distributes 3D radar stations on Earth's surface based on satellite visibility and the Fibonacci lattice.
- **Parameters**:
  - `H_dark`: Maximum satellite altitude in meters.
  - `R_Earth`: Earth's radius.
- **Returns**: Array of shape `(N, 3)` with radar positions `[r, φ, φ]`.

---

```python
def initialise_radar_stations(mode, radar_positions):
    radars = []
    for i, position in enumerate(radar_positions):
        radar = Radar(
            mode=mode,
            ID=f"Radar_{i}",
            location=position,
        )
        radars.append(radar)
    return radars
```
- **Purpose**: Initializes a list of `Radar` objects from a set of radar coordinates.
- **Parameters**:
  - `mode`: Either `'2D'` or `'3D'`.
  - `radar_positions`: Array of radar locations in polar or spherical coordinates.
- **Returns**: A list of `Radar` objects.

---

```python
def weighted_average(x, noise):
    weights = 1 / noise ** 2
    return np.sum(weights * x) / np.sum(weights)
```
- **Purpose**: Computes the weighted average of a set of values using inverse variance weighting.
- **Parameters**:
  - `x`: Array of measurements.
  - `noise`: Array of standard deviations.
- **Returns**: The weighted average value.

---

```python
def combine_radar_measurements(mode, radars, true_traj):
    ...
    if mode.upper() == '3D':
        return np.array([times, r_arr, theta_arr, phi_arr]).T
    else:
        return np.array([times, r_arr, theta_arr]).T
```
- **Purpose**: Aggregates radar measurements at each timestep using weighted averaging of available data.
- **Parameters**:
  - `mode`: Either `'2D'` or `'3D'`.
  - `radars`: List of `Radar` objects.
  - `true_traj`: True trajectory of the satellite.
- **Implementation**:
  - Loops over each timestep and collects visible measurements.
  - Performs weighted averaging using the `weighted_average()` function.
  - If no radar sees the satellite, sets measurement to NaN.
- **Returns**: 2D or 3D array `[time, r, θ (, φ)]` depending on mode.

---

### Radar Class
- `Radar`: Represents a radar station that can track satellite positions, determine visibility, and simulate noisy measurements.

---

```python
def __init__(self, mode, ID, location):
    ...
```
- **Parameters**: 
    - `mode`: `2D` or `3D`.
    - `ID`: The id of radar.
    - `location`: The location of radar.

---

```python
def get_ID(self)
def get_location(self)
def get_visibility_angle(self)
def get_noise(self)
```
- **Purpose**: Standard getter methods to access internal properties.

---

```python
def distance(self, mode, u, v):
    ...
    return np.linalg.norm(cart_u - cart_v, axis=0)
```
- **Purpose**: Computes the Euclidean distance between two points in either 2D or 3D coordinates.
- **Parameters**:
  - `mode`: Either `'2D'` or `'3D'`.
  - `u`, `v`: Coordinate vectors.
- **Returns**: Scalar distance value.

---

```python
def check_visibility(self, satellite_position):
    ...
    return cos_angle_difference >= np.cos(self.__visibility_angle)
```
- **Purpose**: Determines whether a satellite is visible from the radar station, based on geometric constraints.
- **Parameters**:
  - `satellite_position`: Position vector `[r, θ]` or `[r, θ, φ]`.
- **Returns**: Boolean indicating visibility.
- **Implementation**:
  - Converts both radar and satellite position to Cartesian.
  - Uses dot product to compute angle between radar normal and satellite direction.

---

```python
def record_satellite(self, time, satellite_position):
    ...
```
- **Purpose**: Records a satellite position (if visible) at a given time.
- **Parameters**:
  - `time`: Timestamp.
  - `satellite_position`: Satellite position in polar/spherical coordinates.
- **Implementation**:
  - Checks visibility and stores position.
  - If not visible, stores NaNs.

---

```python
def add_noise(self):
    ...
```
- **Purpose**: Adds Gaussian noise to the satellite’s radial distance measurements based on radar-satellite distance.
- **Implementation**:
  - Computes dynamic noise level based on distance using `σ = σ₀ + k·d`.
  - Applies normal-distributed noise to the `r` component.
  - Saves the generated noise.
