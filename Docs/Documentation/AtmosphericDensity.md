# Atmospheric Density

This function provides a utility function for calculating and retrieving atmospheric density using the US Standard Atmosphere 1976 model (USSA1976).

```python
import numpy as np
import ussa1976
ds = ussa1976.compute()
```
- `ussa1976` is a third-party or custom module that computes and provides data based on the 1976 Standard Atmosphere model.
- The function `ussa1976.compute()` returns a data structure that includes physical quantities (e.g., temperature, density, pressure) at various altitudes.

----

```python
def atmos_ussa1976_rho(altitude):
    # Find the closest index to altitude in metres
    idx = np.argmin(np.abs(ds["z"].values - altitude))

    # Return pressure, temperature, and density
    return ds["rho"].values[idx]
```
- **Purpose**: Given an altitude (in meters), it returns the standard atmospheric density (rho).
- **Parameters**: 
    - `altitude`: A float or integer in meters representing the altitude at which to query atmospheric density.
- **Implementation**:
    - Computes the absolute difference between the input altitude and all standard altitude levels in the dataset.
    - Uses `np.argmin()` to find the index of the closest altitude.
    - Returns the density value corresponding to that altitude index.
