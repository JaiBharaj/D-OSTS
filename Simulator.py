import numpy as np
from CrudeInitialConditions import InitialConditions as IC
from NumericalIntegrator import Integrator
from RadarModule import distribute_radars2D, initialise_radar_stations, combine_radar_measurements
from WriteToFiles import write_to_file_2d


mode = '2d'
H_dark = 20000

########## GENERATING TRUE TRAJECTORY ##########
input_path = f"Trajectories/{IC.index}_2d_true_trajectory.txt"
output_path = f"Trajectories/{IC.index}_2d_noisy_trajectory.txt"

rk = Integrator(np.linspace(0,6000, 6001))
true_traj = rk.get_trajectory_2d(input_path)

# optional
write_to_file_2d(input_path, true_traj)

########## RADAR STATION NOISY MEASUREMENTS ##########
H_dark = 20000  # Possible for no radars to see satellite below this height
radar_positions = distribute_radars2D(H_dark, IC.earthRadius)

# Initialise radar stations
radars = initialise_radar_stations(mode, radar_positions)

# Record satellite positions in each radar
for time, r, theta in true_traj:
    sat_pos = [r, theta]
    for radar in radars:
        radar.record_satellite(time, sat_pos)

# Add measurement noise
for radar in radars:
    radar.add_noise()

# Combine measurements from all radars
noisy_traj = combine_radar_measurements(mode, radars, true_traj)
write_to_file_2d(output_path, noisy_traj)
