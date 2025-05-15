import numpy as np

def _to_numpy(*args):
    # Ensure inputs are NumPy arrays (handles scalars and lists)
    return [np.asarray(a) for a in args]


def cartesian_to_polar(x, y):
    x, y = _to_numpy(x, y)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    r, theta = _to_numpy(r, theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cartesian_to_spherical(x, y, z):
    x, y, z = _to_numpy(x, y, z)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)                  # azimuthal angle
    phi = np.arccos(z / np.where(r != 0, r, 1))  # polar angle (avoid div by zero)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    r, theta, phi = _to_numpy(r, theta, phi)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z
