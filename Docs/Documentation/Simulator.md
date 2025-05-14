```python
def run_simulator(mode, H_dark=20_000, recorded_times=np.linspace(0, 6000, 6001), radar_angle=np.pi/2,
                  radar_noise_base=100, radar_noise_scalefactor=0.05, input_path=None, output_path=None)
```

- **Purpose**:

Simulates the motion of a satellite and generates:

1. Ground-truth trajectory via numerical integration.
2. Noisy radar measurements using simulated radar stations.

Outputs are written to files for later use in prediction and visualization modules.

---

- **Parameters**:

* **`mode`** (`str`):
  Simulation mode, determines which variant of the trajectory generation and radar system to use. Typically `'1D'`, `'2D'`, or `'3D'`.

* **`H_dark`** (`float`, default=`20000`):
  Altitude of radar stations above Earth's surface (in meters).

* **`recorded_times`** (`np.ndarray`):
  Array of timestamps at which the satellite's true position is computed (e.g., 0 to 6000 seconds at 1-second intervals).

* **`radar_angle`** (`float`, default=`π/2`):
  Field-of-view angle of each radar station (in radians), used in visibility checks.

* **`radar_noise_base`** (`float`, default=`100`):
  Base level of Gaussian noise to add to radar measurements.

* **`radar_noise_scalefactor`** (`float`, default=`0.05`):
  Factor scaling noise based on distance or angle.

* **`input_path`** (`str` or `None`):
  Path to write the ground-truth (noise-free) trajectory data.

* **`output_path`** (`str` or `None`):
  Path to write the noisy radar measurement data.

---


1. **Generate True Trajectory**:

   * Uses `NumericalIntegrator.Integrator` with the selected mode.
   * Simulates the satellite's physical trajectory in polar or spherical space.
   * Writes to `input_path` using a mode-specific file writer (e.g., `write_to_file_2D`).

2. **Setup Radar Stations**:

   * Calls `distribute_radarsMODE()` (e.g., `distribute_radars3D`) to position radars.
   * Initializes radar objects with angle-of-view and noise characteristics.

3. **Simulate Radar Measurements**:

   * Each radar records satellite positions at given times if within view.
   * Measurement noise is added to simulate real-world uncertainty.

4. **Aggregate and Write Noisy Data**:

   * Combines radar data into a unified noisy trajectory.
   * Writes to `output_path` using the same file writer.

---
