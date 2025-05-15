# Tutorial
This section will establish an understanding of the pre-requisite resources and 
basic workflow for applications created with the De-Orbiting Satellite Tracking 
System (D-OSTS).

## Quickstart Guide
### Installation

First we can try a simple `pip install`.

```bash
pip install dosts
```

If this doesn't work, then try the following:

---

First, after downloading the source code from the D-OSTS GitHub Repository, ensure 
build tools is installed on your local machine. You can easily do this with 
`pip install`.

```bash
pip install build
````

You can then create a `dist/` folder with `.whl` and `.tar.gz` files by using these 
new build tools.

```bash
python -m build
```

Once again, using `pip install`, we locally install D-OSTS.

```bash
pip install dist/dosts-1.0.0-py3-none-any.whl
```

And, that's it, you're ready to start!

---

#### Dependencies
- Works for `Python 3.8` and above,
- `NumPy 1.17.3` and above,
- `Matplotlib 3.1.2` and above,
- `SciPy 1.4.0` and above,
- and `ussa1976 0.3.4`, the lastest version.

It is advised to use the latest versions of each respective package. You may be 
unfamiliar with `ussa1976`, but it can be easily installed with `pip`.

```bash
pip install ussa1976
```

---

### Environment
Test the environment to see if D-OSTS has installed correctly.

From the interactive shell, this is done simply.

```python
>>> import sys
>>> 'dosts' in sys.modules
True
```

Alternatively, successful installation can also be tested in-script.

```python
try:
    import dosts
    print("dosts imported successfully")
except ImportError:
    print("dosts not installed or failed to import")

```

### License
MIT License

Copyright (c) 2025 Jai Bharaj, Tom Miller, David Dukov, Linan Yang, Kunlin Cai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Simulation
### True Trajectory
Generating the true trajectory is easy - we simply initialise the integrator and 
use `get_trajectory`. We begin with the equatorial case first.

```python
from dosts import NumericalIntegrator, WriteToFiles

# Grab the RK45 Simulation Integrator
rk = NumericalIntegrator.Integrator()

# Specify true trajectory path
input_path = "my_true_trajectory_2d.txt"

# Generate trajectory
true_traj = rk.get_trajectory_2d()
WriteToFiles.write_to_file_2d(input_path, true_traj) # Add trajectory data to input file
```

This should generate the equatorial trajectory and save it to the file `"my_true_trajectory_2d.txt"`. 
It should be noted that the initial conditions for this trajectory are stored in 
`dostos.CrudeInitialConditions.InitialConditions` and are easily accessible for setting 
different starting conditions.

For the non-equatorial case, we follow a similar regime.

```python
import numpy as np

# Grab the RK45 Simulation Integrator
rk = NumericalIntegrator.Integrator(recorded_times=np.linspace(0, 6000, 6001))

# Specify true trajectory path
input_path = "my_true_trajectory_3d.txt"

# Generate trajectory
true_traj = rk.get_trajectory_3d()
WriteToFiles.write_to_file_3d()d(input_path, true_traj) # Add trajectory data to input file
```

Additional Note: `record_times` is a variable used in both `get_trajectory_2d()` 
and `get_trajectory_3d()`. This allows us to set specific 'times' at which we output the 
trajectory. For a full trajectory, this can be omitted (as in the first example).

For use of the 'thrust' functionality, we generate 3D trajectory as before, but this time with 
specification of thrust (via bonus).

```python
# Generate trajectory
true_traj = rk.get_trajectory_3d(bonus=True)
```

This tells the integrator to use its internal functions to determine if an expected crash site 
is populated, and engage thrust in the current direction of motion to avoid it.

### Noisy Measurements
To generate the noisy radar measurements we want to import our initial conditions class, 
define our dark zone in which radar stations (generally) are unable to take measurements, 
initialise the radar stations and their positions, and use their in-built functionality 
to use the already generated true trajectory to generate realistic radar data.

```python
from dosts import CrudeInitialConditions, RadarModule, WriteToFiles

IC = CrudeInitialConditions.InitialConditions # Grab initial condition class
mode = '2d' # or '3d' for non-equatorial case

output_path = "my_noisy_trajectory.txt"

# Initialise Radar stations
H_dark = 20000  # Still possible for no radars to see satellite below this height
radar_positions = RadarModule.distribute_radars2D(H_dark, IC.earthRadius) # distribute_radars3D for non-equatorial
radars = RadarModule.initialise_radar_stations(mode, radar_positions)

# Record satellite positions in each radar
    for measurement in true_traj:
        sat_pos = measurement[1:]
        for radar in radars:
            radar.record_satellite(measurement[0], sat_pos)

    # Add measurement noise
    for radar in radars:
        radar.add_noise()

    # Combine measurements from all radars and write to file
    noisy_traj = RadarModule.combine_radar_measurements(mode, radars, true_traj)
    WriteToFiles.write_to_file_2d(output_path, noisy_traj) # write_to_file_3d for non-equatorial
```

The distribution function handles the placement of the radar stations such that there is full coverage above 
`H_dark`. See the documentation for a further review.
### Full Simulation Build
We can now write a full build for the simulator. We'll show how this 
can be done as a reusable function that works for both the equatorial and 
non-equatorial case.

NOTE: The following function is not callable from `dosts`, this is only 
a demonstration of how the simulator can be setup in your environment.

```python
import numpy as np
from dosts import (CrudeInitialConditions,
                   NumericalIntegrator,
                   RadarModule,
                   WriteToFiles)

IC = CrudeInitialConditions.InitialConditions
Integrator = NumericalIntegrator.Integrator

def run_simulator(mode, recorded_times=None):
    input_path = f"Trajectories/{mode}_true_trajectory.txt"
    output_path = f"Trajectories/{mode}_noisy_trajectory.txt"
    write_to_file = getattr(WriteToFiles, f"write_to_file_{mode}")

    ### GET TRUE TRAJECTORY ###
    rk = Integrator(recorded_times)
    get_trajectory = getattr(rk, f"get_trajectory_{mode}")
    true_traj = get_trajectory()
    write_to_file(input_path, true_traj)

    ### NOISY RADAR MEASUREMENTS ###
    H_dark = 200000  # Possible for no radars to see satellite below this height
    distribute_radars = getattr(RadarModule, f"distribute_radars{mode.upper()}")
    radar_positions = distribute_radars(H_dark, IC.earthRadius)

    # Initialise radar stations
    radars = RadarModule.initialise_radar_stations(mode, radar_positions)

    # Record satellite positions in each radar
    for measurement in true_traj:
        sat_pos = measurement[1:]
        for radar in radars:
            radar.record_satellite(measurement[0], sat_pos)

    # Add measurement noise
    for radar in radars:
        radar.add_noise()

    # Combine measurements from all radars and write to file
    noisy_traj = RadarModule.combine_radar_measurements(mode, radars, true_traj)
    write_to_file(output_path, noisy_traj)

######## GENERATING TRUE AND NOISY TRAJECTORY ##########
run_simulator('2d') # Use '3d' for non-equatorial
```

---

## Prediction
### Initialisation
To use our Extended Kalman Filter (EKF) based predictor, we first need 
to initialise the initial conditions to push into them.

```python
from dosts.CrudeInitialConditions import InitialConditions as IC

# file names
input_file = f"Trajectories/2d_noisy_trajectory.txt"
output_file = f"Trajectories/2d_pred_trajectory.txt"
crash_heatmap_file = f"Trajectories/2d_crash_heatmap_data.txt"

# Initial parameters
r0 = IC.earthRadius + IC.initSatAlt
theta0 = IC.initSatTheta

CD = IC.dragCoeff
A = IC.crossSec
m = IC.satMass
GM = IC.gravConstant * IC.earthMass
```

We now want to set up our measurement model, process noise matrix, and initial uncertainty for use in the EKF. 
Take note of how these structures are initialised, but also consider that 
the specific values assigned to each element is completely dependent on assumptions 
you make about the expected noise on radar measurements in a real setting. For a more in-depth 
explanation of how these elements should be set, refer to the documentation or, alternatively, 
the available report.

We begin with a simple equatorial case.

```python
# Measurement noise
sigma_r, sigma_theta = 100.0, 0.0001      # meters, radians

# Measurement model
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

R = np.diag([sigma_r**2, sigma_theta**2])

# Process noise
Q = np.diag([1000, 0, 1e-7, 0])

# Initial uncertainty
P0 = np.diag([
    10.0**2,        # r
    1.0**2,         # vr
    (1e-4)**2,      # theta
    (1e-4)**2       # omega
])
```

This can become increasingly more complex for the non-equatorial case, but we will keep it 
simple by showcasing an example where we assume zero measurement noise on the angular velocities.

```python
# Measurement noise
sigma_r, sigma_theta, sigma_phi = 100.0, 0.0001, 0.0001      # meters, radians, radians

# Measurement model
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

R = np.diag([sigma_r**2, sigma_theta**2, sigma_phi**2])

# Process noise
Q = np.zeros((6, 6))
Q[0, 0] = 100**2
Q[2, 2] = 1e-4**2
Q[4, 4] = 1e-4**2
Q[2, 4] = Q[4, 2] = 0.5 * 1e-4**2

# Initial uncertainty
P0 = np.diag([100.0, 1.0, 1e-8, 1e-8, 1e-8, 1e-8])
```

Now that we have this established, we need to set the initial state to pass into the EKF. 
This is a known quantity ahead of time and is crucial for setting our predictor up to propagate 
properly. Additionally, we need to compute the Jacobian matrix to be passed into the EKF, 
but we have functions that can do this for you for both equatorial and non-equatorial cases.

We start by first showing the equatorial case.

```python
from dosts.AtmosphericDensity import atmos_ussa1976_rho
from dosts.ModelDynamics import PolarAccelerations
from dosts.ExtendedKalmanFilters import compute_F_analytic

# Model dynamics
rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
f_jacobian = lambda x: compute_F_analytic(x, CD, A, m, GM, rho_func=rho_func)
f_dynamics = lambda x: PolarAccelerations.accelerations(x[0], x[1], x[2], x[3])
x0 = np.array([r0, 0.0, theta0, np.sqrt(GM / r0) / r0])
```

Setting up the model dynamics isn't much different for the non-equatorial case. Make sure to 
take note of the change in Jacobian claculation functions.

```python
from dosts.ModelDynamics import SphericalAccelerations
from dosts.ExtendedKalmanFilters import compute_F_spherical

# Model dynamics
rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
f_jacobian = lambda x: compute_F_spherical(x, CD, A, m, GM, rho_func)
f_dynamics = lambda x: SphericalAccelerations.accelerations(*x)

x0 = np.array([r0,
               IC.initSatRdot,
               IC.initSatPhi,
               IC.initSatPhidot,
               IC.initSatLam,
               IC.initSatLamdot])
```

We can now set up the EKF and data containers just before running our prediction loop. This is a 
simple process in that we only need to input our initialised structures, and it doesn't 
vary much between the equatorial and non-equatorial cases.

As usual, lets start with the equatorial.

```python
from dosts.ExtendedKalmanFilters import ExtendedKalmanFilter
from dosts.PredictorIntegrator import RK45Integrator

# Load radar data
data = np.loadtxt(input_file)
measurement_times = data[:, 0]
measurements = data[:, 1:]

# Instantiate EKF and simulate the predicted trajectory and uncertainty
ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    integrator=RK45Integrator(CD, A, m, GM, rho_func)
)

# Reset files
with open(crash_heatmap_file, 'w') as f:  # Clear existing file
    f.write("")
with open(output_file, 'w') as f:  # Clear existing file
    f.write("")
```

And the non-equatorial case has a very similar set up - just take not in the type of 
integrator we import (`Integrator3D()` is generally better for a general non-equatorial case). 
Additionally, take not of the thruster heatmap file; we want to use the in-built functionality 
for not only predicting the crash site, but also the crash site if a thruster is used.

```python
from dosts.PredictorIntegrator import Integrator3D

ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H, Q=Q, R=R,
    x0=x0,
    P0=P0,
    integrator=Integrator3D()
)
########## EKF EXECUTION ##########
with open(crash_heatmap_file, 'w') as f:
    f.write("")
with open(thrust_crash_heatmap_file, 'w') as f:
    f.write("")
with open(output_file, 'w') as f:
    f.write("")

data = np.loadtxt(input_file)
measurement_times = data[:, 0]
measurements = data[:, 1:4]
```

This completes the initialisation phase, we now want to understand how we can predict the trajectory, crash site, and 
thrust-assisted crash site, in a prediction loop.

### Extended Kalman Filters
The purpose of the prediction loop is to be able to take 'incoming' noisy radar measurements, 
and produce an outputted trajectory for the satellite that becomes more accurate with 
with more (appropriate) data. Below we see how to set this up for the equatorial case, 
and how we can use the methods of the EKF class for predicting trajectories and crash sites.

```python
states, covariances, times, is_measured_flags = [], [], [], []
crash_means, crash_stds = [], []

for i, (t, z) in enumerate(zip(measurement_times, measurements)):
    dt = 1e-3 if i == 0 else t - measurement_times[i - 1]
    x, P = ekf.predict(dt)

    is_measured = False
    if not np.isnan(z).any():
        x, P = ekf.update(z)
        is_measured = True

    times.append(t)
    states.append(x.copy())
    covariances.append(P.copy())
    is_measured_flags.append(is_measured)

    if i % 100 == 0:
        crash_angles = ekf.crash(N=10, max_steps=4000)
        print(f"Time {t:.1f}s: {len(crash_angles)} crash predictions")
        if len(crash_angles) > 0:
            # Write timestamp followed by angles
            with open(crash_heatmap_file, 'a') as f:
                f.write(f"{t:.6f} " + ' '.join(f"{angle:.6f}" for angle in crash_angles) + '\n')
```

Take note of how the crash sites are being added to the file. The order of angles is important for 
the non-equatorial case especially as the visualiser cannot accurately display the 
converging sites if given the wrong order. Lets have a look at how we do this:

```python
states, covariances, times, is_measured_flags = [], [], [], []
crash_theta_means, crash_phi_means = [], []
crash_theta_stds, crash_phi_stds = [], []

crash_theta_means_thrust, crash_phi_means_thrust = [], []
crash_theta_stds_thrust, crash_phi_stds_thrust = [], []

delta_v = 5000.0  # m/s
h_thrust = IC.hThrust  # m

log_path = "Trajectories/thrust_decision_log.txt"
with open(log_path, 'w') as f:
    f.write("Time\tp_pop\tp_pop_thrust\tThrustDecision\n")

for i, (t, z) in enumerate(zip(measurement_times, measurements)):
    dt = 1e-3 if i == 0 else t - measurement_times[i - 1]
    x, P = ekf.predict(dt)

    is_measured = False
    if not np.isnan(z).any():
        x, P = ekf.update(z)
        is_measured = True

    times.append(t)
    states.append(x.copy())
    covariances.append(P.copy())
    is_measured_flags.append(is_measured)
    
    if i % 100 == 0:
        crash_angles = ekf.crash3D(N=50, max_steps=10000)
        crash_angles_thrust = ekf.crash3D_with_thrust(delta_v=delta_v, h_thrust=h_thrust, N=50, max_steps=10000)

        with open(crash_heatmap_file, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles for angle in pair) + '\n')

        with open(thrust_crash_heatmap_file, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles_thrust for angle in pair) + '\n')
```

Despite the number of containers increasing with the added thrust-assisted crash site predictions, 
the process of setting up the prediction loop is virtually the same.

### Full Prediction Build
Below are examples of how the predictor files can be set up in their entirety. Please note: 
these are only examples, and as such, they are not importable from the package - they 
serve only as a template for the user to follow when running under their own assumptions 
of incoming data.

#### Equatorial
```python
import numpy as np
from dosts.CrudeInitialConditions import InitialConditions as IC
from dosts.AtmosphericDensity import atmos_ussa1976_rho
from dosts.ModelDynamics import PolarAccelerations
from dosts.ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_analytic
from dosts.PredictorIntegrator import RK45Integrator

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########

# file names
input_file = f"Trajectories/2d_noisy_trajectory.txt"
output_file = f"Trajectories/2d_pred_trajectory.txt"
crash_heatmap_file = f"Trajectories/2d_crash_heatmap_data.txt"

# Initial parameters
r0 = IC.earthRadius + IC.initSatAlt
theta0 = IC.initSatTheta

CD = IC.dragCoeff
A = IC.crossSec
m = IC.satMass
GM = IC.gravConstant * IC.earthMass

# Measurement noise
sigma_r, sigma_theta = 100.0, 0.0001      # meters, radians

# Measurement model
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

R = np.diag([sigma_r**2, sigma_theta**2])

# Process noise
Q = np.diag([1000, 0, 1e-7, 0])

# Initial uncertainty
P0 = np.diag([
    10.0**2,        # r
    1.0**2,         # vr
    (1e-4)**2,      # theta
    (1e-4)**2       # omega
])

# Model dynamics
rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
f_jacobian = lambda x: compute_F_analytic(x, CD, A, m, GM, rho_func=rho_func)
f_dynamics = lambda x: PolarAccelerations.accelerations(x[0], x[1], x[2], x[3])
x0 = np.array([r0, 0.0, theta0, np.sqrt(GM / r0) / r0])

# Load radar data
data = np.loadtxt(input_file)
measurement_times = data[:, 0]
measurements = data[:, 1:]

# Instantiate EKF and simulate the predicted trajectory and uncertainty
ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    integrator=RK45Integrator(CD, A, m, GM, rho_func)
)

# Reset files
with open(crash_heatmap_file, 'w') as f:  # Clear existing file
    f.write("")
with open(output_file, 'w') as f:  # Clear existing file
    f.write("")

states, covariances, times, is_measured_flags = [], [], [], []
crash_means, crash_stds = [], []

for i, (t, z) in enumerate(zip(measurement_times, measurements)):
    dt = 1e-3 if i == 0 else t - measurement_times[i - 1]
    x, P = ekf.predict(dt)

    is_measured = False
    if not np.isnan(z).any():
        x, P = ekf.update(z)
        is_measured = True

    times.append(t)
    states.append(x.copy())
    covariances.append(P.copy())
    is_measured_flags.append(is_measured)

    if i % 100 == 0:
        crash_angles = ekf.crash(N=50, max_steps=4000)
        print(f"Time {t:.1f}s: {len(crash_angles)} crash predictions")
        if len(crash_angles) > 0:
            # Write timestamp followed by angles
            with open(crash_heatmap_file, 'a') as f:
                f.write(f"{t:.6f} " + ' '.join(f"{angle:.6f}" for angle in crash_angles) + '\n')

# Save predicted trajectory
with open(output_file, 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r, theta = x[0], x[2]
        r_std = np.sqrt(P[0, 0])
        theta_std = np.sqrt(P[2, 2])
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {r_std:.6f} {theta_std:.8f} {int(measured)}\n")
```

#### Non-Equatorial
```python
import numpy as np
from dosts.CrudeInitialConditions import InitialConditions as IC
from dosts.AtmosphericDensity import atmos_ussa1976_rho
from dosts.ModelDynamics import SphericalAccelerations
from dosts.ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_spherical
from dosts.PredictorIntegrator import Integrator3D

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########

# file names
input_file = f"Trajectories/3d_noisy_trajectory.txt"
output_file = f"Trajectories/3d_pred_trajectory.txt"
crash_heatmap_file = f"Trajectories/3d_crash_heatmap_data.txt"
thrust_crash_heatmap_file = f"Trajectories/3d_thrust_crash_heatmap_data.txt"

# Initial parameters
r0 = IC.earthRadius + IC.initSatAlt
theta0 = IC.initSatTheta
phi0 = IC.initSatPhi

CD = IC.dragCoeff
A = IC.crossSec
m = IC.satMass
GM = IC.gravConstant * IC.earthMass

# Measurement noise
sigma_r, sigma_theta, sigma_phi = 100.0, 0.0001, 0.0001      # meters, radians, radians

# Measurement model
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

R = np.diag([sigma_r**2, sigma_theta**2, sigma_phi**2])

# Process noise
Q = np.zeros((6, 6))
Q[0, 0] = 100**2
Q[2, 2] = 1e-4**2
Q[4, 4] = 1e-4**2
Q[2, 4] = Q[4, 2] = 0.5 * 1e-4**2

# Initial uncertainty
P0 = np.diag([100.0, 1.0, 1e-8, 1e-8, 1e-8, 1e-8])

# Model dynamics
rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
f_jacobian = lambda x: compute_F_spherical(x, CD, A, m, GM, rho_func)
f_dynamics = lambda x: SphericalAccelerations.accelerations(*x)

x0 = np.array([r0,
               IC.initSatRdot,
               IC.initSatPhi,
               IC.initSatPhidot,
               IC.initSatLam,
               IC.initSatLamdot])

ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H, Q=Q, R=R,
    x0=x0,
    P0=P0,
    integrator=Integrator3D()
)
########## EKF EXECUTION ##########
with open(crash_heatmap_file, 'w') as f:
    f.write("")
with open(thrust_crash_heatmap_file, 'w') as f:
    f.write("")
with open(output_file, 'w') as f:
    f.write("")

data = np.loadtxt(input_file)
measurement_times = data[:, 0]
measurements = data[:, 1:4]

states, covariances, times, is_measured_flags = [], [], [], []
crash_theta_means, crash_phi_means = [], []
crash_theta_stds, crash_phi_stds = [], []

crash_theta_means_thrust, crash_phi_means_thrust = [], []
crash_theta_stds_thrust, crash_phi_stds_thrust = [], []

delta_v = 5000.0  # m/s
h_thrust = IC.hThrust  # m

log_path = "Trajectories/thrust_decision_log.txt"
with open(log_path, 'w') as f:
    f.write("Time\tp_pop\tp_pop_thrust\tThrustDecision\n")

for i, (t, z) in enumerate(zip(measurement_times, measurements)):
    dt = 1e-3 if i == 0 else t - measurement_times[i - 1]
    x, P = ekf.predict(dt)

    is_measured = False
    if not np.isnan(z).any():
        x, P = ekf.update(z)
        is_measured = True

    times.append(t)
    states.append(x.copy())
    covariances.append(P.copy())
    is_measured_flags.append(is_measured)

    if i % 100 == 0:
        crash_angles = ekf.crash3D(N=50, max_steps=10000)
        crash_angles_thrust = ekf.crash3D_with_thrust(delta_v=delta_v, h_thrust=h_thrust, N=50, max_steps=10000)

        with open(crash_heatmap_file, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles for angle in pair) + '\n')

        with open(thrust_crash_heatmap_file, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles_thrust for angle in pair) + '\n')

########## SAVE TRAJECTORY ##########
with open(output_file, 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r, theta, phi = x[0], x[2], x[4]
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {phi:.8f} "
                f"{np.sqrt(P[0, 0]):.3f} {np.sqrt(P[2, 2]):.3e} {np.sqrt(P[4, 4]):.3e} {int(measured)}\n")
```

---

NOTE: For a simpler example of a full build, refer to `Main.py`, `Simulator.py`, `Predictor.py`, and 
`MainVisualise.py` in the GitHub Repository. These scripts are more user friendly, but simplified 
in that they lack the functionality for grabbing crash sites and their heatmaps. It's 
recommended to use the scripts provided in this document for full functionality.

---

## Visualisation
Using the visualiser is fairly intuitive. We first need to call the visualiser we'd 
like to use from `dosts.Visualise`, and then we can initialise our visualiser. As always, 
lets begin with the equatorial case.

```python
from dosts.Visualiser import Visualiser2D

true_traj_file = f"Trajectories/2d_true_trajectory.txt"
pred_traj_file = f"Trajectories/2d_pred_trajectory.txt"
crash_heat_file = f"Trajectories/2d_crash_heatmap_data.txt"

vis = Visualiser2D(true_traj_file, pred_traj_file, crash_heat_file, mode='prewritten')
vis.visualise()
```

Note the use of `mode`. If this is set to `'realtime'`, then the visualiser will work 
on the basis that we receive noisy measurements from actual radar stations as a satellite 
is actually de-orbiting. If nothing is set, the visualiser will defualt to `'prewritten'` mode.

The non-equatorial visualiser has the exact same functionality, plus an additional 
feature for the predicted thrust-assisted crash sites.

```python
from dosts.Visualiser import Visualiser3D

true_traj_file = f"Trajectories/3d_true_trajectory.txt"
pred_traj_file = f"Trajectories/3d_pred_trajectory.txt"
crash_heatmap_file = f"Trajectories/3d_crash_heatmap_data.txt"
thrust_crash_heatmap_file = f"Trajectories/3d_thrust_crash_heatmap_data.txt"

vis = Visualiser3D(true_traj_file, pred_traj_file, crash_heatmap_file, thrust_crash_heatmap_file)
vis.visualise()
```

Note that `mode` wasn't set here. As already mentioned, in the event in which no mode is provided, 
the visualiser will default to `'prewritten'`.

In both cases, the visualiser does not require an input for heatmaps. In the event of 
no heatmap data, the visualiser will show a bisual as normal, just without the heatmapping 
visuals. However, an input is always required for the true and predicted trajectories. In 
a realtime event, where we don't have a true trajectory, inputting `None` is expected.

```python
>>> Visualiser3D(true_traj_file, None) # Works!
>>> Visualiser3D(None, pred_traj_file) # Works!

>>> Visualiser3D(true_traj_file) # Does NOT Work!
>>> Visualiser3D(pred_traj_file) # Does NOT Work!
```