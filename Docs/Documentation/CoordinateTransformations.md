# Coordinate Transformations

This script provides a set of utility functions for coordinate transformations between Cartesian, Polar, and Spherical coordinate systems.

----

```python
def _to_numpy(*args):
    return [np.asarray(a) for a in args]
```
- **Purpose**: Converts input arguments into NumPy arrays to ensure consistent calculations.

----

```python
def cartesian_to_polar(x, y):
    x, y = _to_numpy(x, y)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta
```
- **Purpose**: Converts 2D Cartesian coordinates `(x, y)` into polar coordinates `(r, θ)`.

----

```python
def polar_to_cartesian(r, theta):
    r, theta = _to_numpy(r, theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
```
- **Purpose**: Converts polar coordinates `(r, θ)` back into 2D Cartesian coordinates `(x, y)`.

----

```python
def cartesian_to_spherical(x, y, z):
    x, y, z = _to_numpy(x, y, z)
    r = np.sqrt(x**2 + y**2 + z**2)
    # azimuthal angle
    theta = np.arctan2(y, x)
    # polar angle (avoid div by zero)    
    phi = np.arccos(z / np.where(r != 0, r, 1))  
    return r, theta, phi
```
- **Purpose**: Converts 3D Cartesian coordinates `(x, y, z)` into spherical coordinates `(r, θ, φ)`.

----

```python
def spherical_to_cartesian(r, theta, phi):
    r, theta, phi = _to_numpy(r, theta, phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z
```
- **Purpose**: Converts spherical coordinates `(r, θ, φ)` back into 3D Cartesian coordinates `(x, y, z)`.