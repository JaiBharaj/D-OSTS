# Tutorial
This section will establish an understanding of the pre-requisite resources and 
basic workflow for applications created with the De-Orbiting Satellite Tracking 
System (D-OSTS).

## Quickstart Guide
### Installation
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

#### Dependencies
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

## Prediction
### Initialisation
### Extended Kalman Filters
### Full Prediction Build

## Examples
### Equatorial
### Non-Equatorial
### Thruster
