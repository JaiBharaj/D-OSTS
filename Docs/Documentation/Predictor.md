# Predictor

This module defines the main interface to run orbit prediction using Extended Kalman Filtering (EKF). It supports both 2D (polar) and 3D (spherical) modes, reads radar measurement files, and writes out filtered trajectories and uncertainties.

---

```python
def run_predictor(mode, input_file_name, output_file_name, uncertainty_file_name):
    if mode.upper()=='2D':
        run_predictor_2d(input_file_name, output_file_name, uncertainty_file_name)
    elif mode.upper()=='3D':
        run_predictor_3d(input_file_name, output_file_name, uncertainty_file_name)
    else:
        print("Mode unidentified. Choose between '2d' or '3d'.")
```
- **Purpose**: Entry point for the prediction routine. Dispatches to either 2D or 3D prediction function based on mode.
- **Parameters**:
  - `mode`: `'2D'` or `'3D'`.
  - `input_file_name`: Path to radar measurement input file.
  - `output_file_name`: Path to save predicted trajectory.
  - `uncertainty_file_name`: Path to save uncertainty estimates.

---

```python
def run_predictor_2d(input_file_name, output_file_name, uncertainty_file_name):
    ...
```
- **Purpose**: Runs the 2D orbital prediction using an Extended Kalman Filter (EKF) and writes results to file.
- **Parameters**:
  - `input_file_name`: Text file with radar measurements.
  - `output_file_name`: File to write predicted 2D trajectory `[t, r, θ]`.
  - `uncertainty_file_name`: File to write prediction uncertainty.
- **Implementation**:
  - Initializes satellite and atmospheric parameters from `InitialConditions`.
  - Sets up the measurement model `H`, noise matrix `R`, process noise `Q`, and initial uncertainty `P0`.
  - Defines:
    - `f_dynamics`: State propagation function using `PolarAccelerations`.
    - `f_jacobian`: Jacobian function via `compute_F_analytic`.
  - Loads radar data and runs `ExtendedKalmanFilter`.
  - Writes both predicted state and uncertainty to file.

---

```python
def run_predictor_3d(input_file_name, output_file_name, uncertainty_file_name):
    ...
```
- **Purpose**: Runs the 3D orbital prediction using EKF with spherical dynamics and writes results to file.
- **Parameters**:
  - `input_file_name`: Text file with radar measurements.
  - `output_file_name`: File to write predicted 3D trajectory `[t, r, θ, φ]`.
  - `uncertainty_file_name`: File to write prediction uncertainty.
- **Implementation**:
  - Initializes satellite constants and initial state using `InitialConditions`.
  - Defines:
    - Measurement matrix `H` for 3D.
    - Covariance matrices `R`, `Q`, `P0`.
    - Dynamics via `SphericalAccelerations`.
    - Jacobian via `compute_F_spherical`.
  - Loads data, runs EKF using `Integrator3D` backend.
  - Writes predictions and uncertainties to output files.
