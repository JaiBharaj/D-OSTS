import numpy as np
from dosts import (CrudeInitialConditions,
                   NumericalIntegrator,
                   RadarModule,
                   WriteToFiles)

IC = CrudeInitialConditions.InitialConditions
Integrator = NumericalIntegrator.Integrator

def run_simulator(mode, recorded_times=None):
    input_path = f"Trajectories/{mode}_1kt_true_trajectory.txt"
    output_path = f"Trajectories/{mode}_1kt_noisy_trajectory.txt"
    write_to_file = getattr(WriteToFiles, f"write_to_file_{mode}")

    ### GET TRUE TRAJECTORY ###
    rk = Integrator(recorded_times)
    get_trajectory = getattr(rk, f"get_trajectory_{mode}")
    true_traj = get_trajectory(bonus=True)
    write_to_file(input_path, true_traj)

    ### NOISY RADAR MEASUREMENTS ###
    H_dark = 200000  # Possible for no radars to see satellite below this height
    distribute_radars = getattr(RadarModule, f"distribute_radars{mode.upper()}")
    radar_positions = distribute_radars(H_dark, IC.earthRadius)
    radar_angle = np.pi / 2
    radar_noise_base = 50  # meters
    radar_noise_scalefactor = 0.0005  # m/km

    # Initialise radar stations
    radars = RadarModule.initialise_radar_stations(mode,
                                                   radar_positions,
                                                   radar_angle,
                                                   radar_noise_base,
                                                   radar_noise_scalefactor)

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
run_simulator('3d')