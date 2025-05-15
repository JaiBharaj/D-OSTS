import numpy as np
import ussa1976

ds = ussa1976.compute()

def atmos_ussa1976_rho(altitude):
    # Find the closest index to altitude in metres
    idx = np.argmin(np.abs(ds["z"].values - altitude))

    # Return pressure, temperature, and density
    return ds["rho"].values[idx]

# print(atmos_ussa1976_rho(1000, ds))