import numpy as np
from dosts import CrudeInitialConditions, NumericalIntegrator, RadarModule, WriteToFiles

IC = CrudeInitialConditions.InitialConditions
Integrator = NumericalIntegrator.Integrator

def run_simulator(mode, recorded_times=np.linspace(0, 6000, 6001)):
    input_path = f"Trajectories/{mode}_{IC.index}_true_trajectory.txt"
    output_path = f"Trajectories/{mode}_{IC.index}_noisy_trajectory.txt"
    write_to_file = getattr(WriteToFiles, f"write_to_file_{mode}")

    ### GET TRUE TRAJECTORY ###
    rk = Integrator(recorded_times)
    get_trajectory = getattr(rk, f"get_trajectory_{mode}")
    true_traj = get_trajectory()
    write_to_file(input_path, true_traj)

    ### NOISY RADAR MEASUREMENTS ###
    H_dark = 20000  # Possible for no radars to see satellite below this height
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

run_simulator('3d')

